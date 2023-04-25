from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification


@lru_cache(3)
def get_lm(lm_name):
    return AutoModelForCausalLM.from_pretrained(lm_name)


@lru_cache(3)
def get_cls(cls_name):
    return AutoModelForSequenceClassification.from_pretrained(cls_name)
