import torch
import pandas as pd
from scipy.special import softmax


def filter_paraphrases(
        paraphrase_func,
        paraphrase_kwargs: dict,
        filter_cls_model,
        filter_tokenizer,
        target_cls_id=0,
        cls_score_threshold=0.50,
        max_tries=10
):
    num_tries = 0
    best_cls_score = 0.0
    para_texts = []
    for i in range(max_tries):
        para_text = paraphrase_func(**paraphrase_kwargs)
        cls_encoded_input = filter_tokenizer(para_text, return_tensors='pt').to(filter_cls_model.device)
        cls_output = filter_cls_model(**cls_encoded_input)
        cls_probas = softmax(cls_output.logits[0].cpu().detach().numpy(), axis=0)
        cls_score = cls_probas[target_cls_id]
        para_texts.append((para_text, cls_score))
        num_tries += 1

        if cls_score > best_cls_score:
            best_para_text = para_text
            best_cls_score = cls_score
        if cls_score > cls_score_threshold:
            break

    return {
        'best_para_text': (best_para_text, best_cls_score),
        'para_texts': para_texts
    }


def convert_results_to_df(
        results: dict
):
    results_df = pd.DataFrame()

    results_df['best_para_text'] = [r['best_para_text'][0] for r in results]
    results_df['num_tries'] = [len(r['para_texts']) for r in results]
    results_df['best_score'] = [r['best_para_text'][1] for r in results]

    return results_df
