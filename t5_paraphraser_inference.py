from caif.inference import caif_inference
from transformers import T5Tokenizer


def t5_paraphrase(text,
                    model,
                    tokenizer,
                    beams=5,
                    grams=4,
                    do_sample=False,
                    top_k=20,
                    num_samples=1,
                    bad_words_ids=None
                    ):
    texts = [text] if isinstance(text, str) else text
    x = tokenizer(texts, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(
        **x,
        encoder_no_repeat_ngram_size=grams,
        num_beams=beams,
        max_length=max_size,
        top_k=top_k,
        do_sample=do_sample,
        num_return_sequences=num_samples,
        bad_words_ids=bad_words_ids
    )
    result = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
    if len(result) == 1:
        return result[0]
    else:
        return result


def caif_t5_paraphrase(
        text,
        lm_model_name,
        cls_model_name,
        fp16=False,
        alpha=-5,
        target_label_id=1,
        entropy_threshold=0,
        encoder_no_repeat_ngram_size=None,
        num_samples=1,
        act_type='sigmoid'
):
    tokenizer = T5Tokenizer.from_pretrained(lm_model_name)

    max_length = int(len(tokenizer.encode(text)) * 1.5 + 10)
    output_sequences = caif_inference(
        lm_model_name=lm_model_name,
        cls_model_name=cls_model_name,
        paraphraser_lm=False,
        prompt=text,
        fp16=fp16,
        alpha=alpha,
        target_label_id=target_label_id,
        entropy_threshold=entropy_threshold,
        act_type=act_type,
        num_tokens=max_length,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        num_samples=num_samples
    )

    if len(output_sequences) == 1:
        return output_sequences[0]
    else:
        return output_sequences
