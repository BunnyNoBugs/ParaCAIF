import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import clean
from .caif.inference import caif_inference


def rp_gpt_paraphrase(
        text,
        model,
        tokenizer,
        temperature=1,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        stop_token='</s>'
):
    prompt = f'<s>{text} === '
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(model.device)
    max_length = int(encoded_prompt.shape[1] * 1.5 + 10)
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences
    )
    output_text = tokenizer.decode(output_sequences[0])
    output_text = output_text.replace('=== ===', '===')  # todo: move to utils.clean
    output_text = output_text[: output_text.find(stop_token) if stop_token else None]
    output_text = output_text[len(tokenizer.decode(encoded_prompt[0])):]
    output_text = clean(output_text)

    return output_text


def caif_rp_gpt_paraphrase(
        text,
        lm_model_name,
        cls_model_name,
        fp16=True,
        alpha=-5,
        target_label_id=1,
        entropy_threshold=0,
        act_type='sigmoid',
        stop_token='</s>'
):
    tokenizer = GPT2Tokenizer.from_pretrained(lm_model_name)

    prompt = f'<s>{text} === '
    max_length = int(len(tokenizer.encode(prompt)) * 1.5 + 10)
    output_text = caif_inference(
        lm_model_name=lm_model_name,
        cls_model_name=cls_model_name,
        prompt=prompt,
        fp16=fp16,
        alpha=alpha,
        target_label_id=target_label_id,
        entropy_threshold=entropy_threshold,
        act_type=act_type,
        num_tokens=max_length
    )
    output_text = output_text.replace('=== ===', '===')
    output_text = output_text[: output_text.find(stop_token) if stop_token else None]
    output_text = output_text[len(prompt):]
    output_text = clean(output_text)

    return output_text
