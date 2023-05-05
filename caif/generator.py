from typing import Optional, Union

import torch
import transformers

from .utils import get_lm


class Generator:
    def __init__(self, lm_model_name, device, entropy=None):

        self.device = device

        if 't5' in lm_model_name:
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(lm_model_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                lm_model_name
            )
        self.lm = get_lm(lm_model_name).to(device)
        self.lm.eval()

        self.lm.config.pad_token_id = self.lm.config.eos_token_id
        self.tokenizer.add_special_tokens(
            {"pad_token": self.tokenizer.decode(self.lm.config.eos_token_id)}
        )
        self.caif_sampler = None
        self.ordinary_sampler = None
        self.entropy_based_stats = {
            "skips": 0,
            "avg_entropy": 0,
            "count": 0,
        }
        self.input_prompt_ids = None
        self.entropy = entropy

    def set_caif_sampler(self, sampler):
        self.caif_sampler = sampler

    def set_ordinary_sampler(self, sampler):
        self.ordinary_sampler = sampler

    def sample_sequences(
        self,
        num_samples: int,
        input_prompt: Optional[str],
        max_length: int,
        caif_period: int,
        caif_tokens_num: Union[int, None] = None,
        entropy: float = None,
        progress_bar=None,
        **sampler_kwargs
    ):
        self.input_prompt_ids = self.tokenizer(input_prompt, return_tensors='pt').input_ids.to(self.device)
        self.entropy = entropy

        input_ids, past, ended_sequences = self.get_input_ids(
            input_prompt,
            num_samples,
        )
        gen_history = []
        inp_len = len(input_ids[0])
        if self.caif_sampler is not None:
            current_decoded = self.tokenizer.decode(input_ids[0])
            probs = torch.exp(
                self.caif_sampler.get_classifier_log_probs(
                    current_decoded, target_cls_id=sampler_kwargs["target_cls_id"]
                )
            ).item()
            gen_history += [probs]
        for i in range(max_length):
            is_caif_step = (
                i % caif_period == 0 and self.caif_sampler is not None
            )
            input_ids, past, ended_sequences = self.generation_step(
                input_ids,
                past,
                ended_sequences,
                is_caif_step,
                caif_tokens_num=caif_tokens_num,
                **sampler_kwargs
            )
            if progress_bar:
                progress_bar.progress((i+1)/max_length)
            if ended_sequences.all():
                break
            current_decoded = self.tokenizer.decode(input_ids[0])
            if self.caif_sampler is not None:
                probs = torch.exp(
                    self.caif_sampler.get_classifier_log_probs(
                        current_decoded, target_cls_id=sampler_kwargs["target_cls_id"]
                    )
                ).item()
                gen_history += [probs]

        return (
            [
                self.tokenizer.decode(sequence, skip_special_tokens=True)
                for sequence in input_ids
            ],
            input_ids,
        )

    def generation_step(
        self,
        input_ids,
        past,
        ended_sequences,
        is_caif_step: bool,
        caif_tokens_num=None,
        **sampler_kwargs
    ):
        if self.lm.config.is_encoder_decoder:
            outputs = self.lm.generate(
                input_ids=self.input_prompt_ids,
                decoder_input_ids=input_ids,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
            outputs.logits = outputs.scores[-1].unsqueeze(0)
            outputs.past_key_values = None
        else:
            prepared_inputs = self.lm.prepare_inputs_for_generation(
                input_ids, past, use_cache=True
            )
            outputs = self.lm(
                **prepared_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )

        past = outputs.past_key_values
        if self.entropy is not None:
            normalized = torch.nn.functional.log_softmax(
                outputs.logits, dim=-1
            )
            p = torch.exp(normalized)
            output_probs = p
            output_information = -normalized
            output_entropy = (output_probs * output_information).sum(-1)[:, -1]
            batch_size = output_entropy.shape[0]
            caif_mask = torch.ge(output_entropy, self.entropy)
            ordinary_mask = ~caif_mask
            self.entropy_based_stats["skips"] += caif_mask.sum() / batch_size
            self.entropy_based_stats["count"] += 1
            self.entropy_based_stats["avg_entropy"] += (
                output_entropy.sum() / batch_size
            )
            flatten_entropy = output_entropy.view(-1).cpu().tolist()
            if "entropy" not in self.entropy_based_stats.keys():
                self.entropy_based_stats["entropy"] = flatten_entropy
            else:
                self.entropy_based_stats["entropy"] += flatten_entropy

            if caif_mask.sum() == 0:
                next_tokens_sampler = self.ordinary_sampler
                next_tokens = next_tokens_sampler(
                    input_ids,
                    outputs.logits,
                    caif_tokens_num=caif_tokens_num,
                    **sampler_kwargs
                )
                next_tokens = (
                    next_tokens * (1 - ended_sequences.long())
                    + self.lm.config.eos_token_id * ended_sequences.long()
                ).long()

            elif caif_mask.sum() == batch_size:
                next_tokens_sampler = self.caif_sampler
                next_tokens = next_tokens_sampler(
                    input_ids,
                    outputs.logits,
                    caif_tokens_num=caif_tokens_num,
                    **sampler_kwargs
                )
                next_tokens = (
                    next_tokens * (1 - ended_sequences.long())
                    + self.lm.config.eos_token_id * ended_sequences.long()
                ).long()

            else:
                next_tokens_caif = self.caif_sampler(
                    input_ids[caif_mask],
                    outputs.logits[caif_mask],
                    caif_tokens_num=caif_tokens_num,
                    **sampler_kwargs
                )
                next_tokens_ordinary = self.ordinary_sampler(
                    input_ids[ordinary_mask],
                    outputs.logits[ordinary_mask],
                    caif_tokens_num=caif_tokens_num,
                    **sampler_kwargs
                )
                next_tokens_caif = (
                    next_tokens_caif * (1 - ended_sequences[caif_mask].long())
                    + self.lm.config.eos_token_id
                    * ended_sequences[caif_mask].long()
                ).long()
                next_tokens_ordinary = (
                    next_tokens_ordinary
                    * (1 - ended_sequences[ordinary_mask].long())
                    + self.lm.config.eos_token_id
                    * ended_sequences[ordinary_mask].long()
                ).long()

                next_tokens = torch.ones(batch_size).long().to(self.device)
                next_tokens[caif_mask] = next_tokens_caif
                next_tokens[ordinary_mask] = next_tokens_ordinary
        else:
            if is_caif_step:
                next_tokens_sampler = self.caif_sampler
            else:
                next_tokens_sampler = self.ordinary_sampler

            next_tokens = next_tokens_sampler(
                input_ids,
                outputs.logits,
                caif_tokens_num=caif_tokens_num,
                **sampler_kwargs
            )

            next_tokens = (
                next_tokens * (1 - ended_sequences.long())
                + self.lm.config.eos_token_id * ended_sequences.long()
            ).long()

        input_ids = torch.cat(
            [input_ids, next_tokens[:, None].to(self.device)], dim=-1
        )

        ended_sequences += next_tokens == self.lm.config.eos_token_id

        return input_ids, past, ended_sequences

    def get_input_ids(self, input_prompt, num_samples):
        if self.lm.config.is_encoder_decoder:
            input_ids = torch.tensor([[self.lm.config.decoder_start_token_id]])
        else:
            if input_prompt is not None:
                input_prompt = self.tokenizer(
                    input_prompt, return_tensors="pt"
                ).input_ids
                input_ids = input_prompt
        input_ids = input_ids.repeat(num_samples, 1).to(self.device)
        past = None
        ended_sequences = torch.zeros(
            input_ids.shape[0], device=self.device
        ).bool()

        return input_ids, past, ended_sequences

    @staticmethod
    def sample(unscaled_probs, values):
        samples = torch.multinomial(unscaled_probs, 1)
        return torch.take_along_dim(values, samples, dim=1)
