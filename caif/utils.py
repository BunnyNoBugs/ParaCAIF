from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, T5ForConditionalGeneration


@lru_cache(3)
def get_lm(lm_name):
    if 't5' in lm_name:
        return T5ForConditionalGeneration.from_pretrained(lm_name)
    else:
        return AutoModelForCausalLM.from_pretrained(lm_name)


@lru_cache(3)
def get_cls(cls_name):
    return AutoModelForSequenceClassification.from_pretrained(cls_name)
