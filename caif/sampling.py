from typing import List

from functools import lru_cache

import torch
from torch.nn import functional as F

import transformers

from .utils import get_cls


def sample_from_values(unscaled_probs, values):
    samples = torch.multinomial(unscaled_probs, 1)
    return torch.take_along_dim(values, samples, dim=1)


class TopKWithTemperatureSampler:
    def __call__(self, input_ids, output_logits, top_k, temperature, **kwargs):

        next_token_logits = output_logits[:, -1]
        next_token_log_probs = F.log_softmax(
            next_token_logits, dim=-1
        )

        topk_log_probs = next_token_log_probs.topk(top_k, -1)
        next_tokens = sample_from_values(
            torch.exp(topk_log_probs[0] / temperature), topk_log_probs[1]
        ).squeeze(1)

        return next_tokens


class CAIFSampler:
    @lru_cache(20)
    def __init__(self, classifier_name, lm_tokenizer, device, invert_cls_probs: bool = False):
        self.device = device
        self.classifier_tokenizer = transformers.AutoTokenizer.from_pretrained(
            classifier_name
        )
        self.classifier_model = (
            get_cls(classifier_name).to(device)
        )
        self.classifier_model.eval()
        self.lm_tokenizer = lm_tokenizer
        self.invert_cls_probs = invert_cls_probs

    def __call__(
        self,
        input_ids,
        output_logis,
        top_k,
        temperature,
        top_k_classifier,
        classifier_weight,
        caif_tokens_num=None,
        act_type: str = "sigmoid",
        target_cls_id: int = 0,
        **kwargs
    ):
        next_token_logits = output_logis[:, -1]
        next_token_log_probs = F.log_softmax(
            next_token_logits, dim=-1
        )

        (next_token_unnormalized_probs, topk_indices,) = self.get_unnormalized_probs(
            input_ids,
            next_token_log_probs,
            temperature,
            top_k_classifier,
            classifier_weight,
            caif_tokens_num=caif_tokens_num,
            target_cls_id=target_cls_id
        )
        topk_probs = next_token_unnormalized_probs.topk(top_k, -1)
        next_tokens = sample_from_values(
            topk_probs[0],
            torch.take_along_dim(topk_indices, topk_probs[1], dim=1),
        ).squeeze(1)

        return next_tokens

    def get_unnormalized_probs(
        self,
        input_ids,
        next_token_log_probs,
        temperature,
        top_k_classifier,
        classifier_weight,
        target_cls_id: int = 0,
        act_type: str = "sigmoid",
        caif_tokens_num=None
    ):

        if classifier_weight == 0.0:
            raise ValueError(
                "classifier weight equal to 0 is not supported for CAIF Sampling"
            )

        top_next_token_log_probs = next_token_log_probs.topk(top_k_classifier, -1)
        classifier_input = torch.cat(
            [
                input_ids.unsqueeze(1).repeat(1, top_k_classifier, 1).flatten(0, 1),
                top_next_token_log_probs[1].view(-1).unsqueeze(-1),
            ],
            -1,
        )
        classifier_input = [
            self.lm_tokenizer.decode(sequence, skip_special_tokens=True)
            for sequence in classifier_input
        ]

        if self.invert_cls_probs:
            classifier_log_probs = torch.log(
                1 - self.get_classifier_probs(
                    classifier_input, caif_tokens_num=caif_tokens_num, target_cls_id=target_cls_id
                ).view(-1, top_k_classifier)
            )
        else:
            classifier_log_probs = self.get_classifier_log_probs(
                classifier_input,
                caif_tokens_num=caif_tokens_num,
                target_cls_id=target_cls_id,
                act_type=act_type,
            ).view(-1, top_k_classifier)

        next_token_probs = torch.exp(
            (top_next_token_log_probs[0] +
             classifier_weight * (classifier_log_probs - classifier_log_probs.mean(-1)) -
             top_next_token_log_probs[0].mean(-1))
            / temperature
        )
        return next_token_probs, top_next_token_log_probs[1]

    def get_classifier_log_probs(self, input, caif_tokens_num=None, target_cls_id: int = 0, act_type: str = "sigmoid"):
        input_ids = self.classifier_tokenizer(
            input, padding=True, return_tensors="pt"
        ).to(self.device)
        if caif_tokens_num is not None:
            input_ids["input_ids"] = input_ids["input_ids"][:, -caif_tokens_num:]
            if "attention_mask" in input_ids.keys():
                input_ids["attention_mask"] = input_ids["attention_mask"][:, -caif_tokens_num:]
            if "token_type_ids" in input_ids.keys():
                input_ids["token_type_ids"] = input_ids["token_type_ids"][:, -caif_tokens_num:]

        if act_type == "sigmoid":
            logits = self.classifier_model(**input_ids).logits[:, target_cls_id].squeeze(-1)
            return F.logsigmoid(logits)
        if act_type == "softmax":
            logits = F.log_softmax(self.classifier_model(**input_ids).logits)[:, target_cls_id].squeeze(-1)
            return logits

    def get_classifier_probs(self, input, caif_tokens_num=None, target_cls_id: int = 0):
        input_ids = self.classifier_tokenizer(
            input, padding=True, return_tensors="pt"
        ).to(self.device)
        if caif_tokens_num is not None:
            input_ids["input_ids"] = input_ids["input_ids"][-caif_tokens_num:]
            if "attention_mask" in input_ids.keys():
                input_ids["attention_mask"] = input_ids["attention_mask"][-caif_tokens_num:]
        logits = self.classifier_model(**input_ids).logits[:, target_cls_id].squeeze(-1)
        return torch.sigmoid(logits)
