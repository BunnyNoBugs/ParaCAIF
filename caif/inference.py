import torch

import transformers

from .sampling import CAIFSampler, TopKWithTemperatureSampler
from .generator import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"


def caif_inference(
        lm_model_name: str,
        cls_model_name: str,
        prompt: str,
        paraphraser_lm: bool = False,
        fp16: bool = True,
        alpha: float = 5,
        target_label_id: int = 0,
        entropy_threshold: float = 0,
        act_type: str = "sigmoid",
        num_tokens=10,
        num_samples=1
) -> str:
    torch.set_grad_enabled(False)
    generator = Generator(lm_model_name=lm_model_name, device=device)
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(lm_model_name)
    if alpha != 0:
        if paraphraser_lm:
            caif_sampler = CAIFSampler(classifier_name=cls_model_name, lm_tokenizer=lm_tokenizer, device=device,
                                       prompt_len=len(prompt))
        else:
            caif_sampler = CAIFSampler(classifier_name=cls_model_name, lm_tokenizer=lm_tokenizer, device=device)
        if entropy_threshold < 0.05:
            entropy_threshold = None
    else:
        caif_sampler = None
        entropy_threshold = None

    generator.set_caif_sampler(caif_sampler)
    ordinary_sampler = TopKWithTemperatureSampler()
    kwargs = {
        "top_k": 20,
        "temperature": 1.0,
        "top_k_classifier": 100,
        "classifier_weight": alpha,
        "target_cls_id": target_label_id,
        "act_type": act_type
    }
    generator.set_ordinary_sampler(ordinary_sampler)
    if device == "cpu":
        autocast = torch.cpu.amp.autocast
    else:
        autocast = torch.cuda.amp.autocast
    with autocast(fp16):
        # print(f"Generating for prompt: {prompt}")
        sequences, tokens = generator.sample_sequences(
            num_samples=num_samples,
            input_prompt=prompt,
            max_length=num_tokens,
            caif_period=1,
            entropy=entropy_threshold,
            **kwargs
        )
        # print(f"Output for prompt: {sequences}")

    return sequences
