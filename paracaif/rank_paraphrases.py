from typing import List
import pandas as pd
from scipy.special import softmax
from scipy.spatial import distance


def rank_paraphrases(
        candidates: List[str],
        origin: str,
        style_cls_model,
        cls_tokenizer,
        sim_model,
        target_cls_id=0,
        style_score_threshold=0.5
):
    candidate_embeddings = sim_model.encode(candidates)
    origin_embedding = sim_model.encode(origin)

    cls_encoded_input = cls_tokenizer(candidates, padding=True, return_tensors='pt').to(style_cls_model.device)
    cls_output = style_cls_model(**cls_encoded_input)
    cls_probas = softmax(cls_output.logits.cpu().detach().numpy(), axis=1)

    scored_candidates = []
    for candidate, candidate_embedding, candidate_cls_probas in zip(candidates, candidate_embeddings, cls_probas):
        sim_score = 1 - distance.cosine(candidate_embedding, origin_embedding)
        style_score = candidate_cls_probas[target_cls_id]
        scored_candidates.append((sim_score, style_score, candidate))
    scored_candidates = sorted(scored_candidates, reverse=True)
    best_style_score = 0
    for candidate in scored_candidates:
        style_score = candidate[1]
        if style_score > best_style_score:
            best_candidate = candidate
            if style_score > style_score_threshold:
                break

    return {
        'best_candidate': best_candidate,
        'ranked_candidates': scored_candidates
    }


def convert_results_to_df(
        results: dict
):
    results_df = pd.DataFrame()

    results_df['best_para_text'] = [r['best_para_text'][0] for r in results]
    results_df['num_tries'] = [len(r['para_texts']) for r in results]
    results_df['best_score'] = [r['best_para_text'][1] for r in results]

    return results_df
