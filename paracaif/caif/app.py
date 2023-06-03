import os
from typing import Tuple

import streamlit as st

import torch

import transformers

from transformers import AutoConfig
import tokenizers

from sampling import CAIFSampler, TopKWithTemperatureSampler
from generator import Generator

import pickle

from plotly import graph_objects as go

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

ATTRIBUTE_MODELS = {
    "English": (
        "distilbert-base-uncased-finetuned-sst-2-english",
        "unitary/toxic-bert",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
}

CITE = """@misc{https://doi.org/10.48550/arxiv.2205.07276,
  doi = {10.48550/ARXIV.2205.07276},
  url = {https://arxiv.org/abs/2205.07276},
  author = {Sitdikov, Askhat and Balagansky, Nikita and Gavrilov, Daniil and Markov, Alexander},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Classifiers are Better Experts for Controllable Text Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

LANGUAGE_MODELS = {
    "English": ("gpt2", "distilgpt2", "EleutherAI/gpt-neo-1.3B")
}

ATTRIBUTE_MODEL_LABEL = {
    "English": "Choose attribute model"
}

LM_LABEL = {
    "English": "Choose language model",
}

ATTRIBUTE_LABEL = {
    "English": "Choose desired attribute",
}

TEXT_PROMPT_LABEL = {
    "English": "Text prompt",
}

PROMPT_EXAMPLE = {
    "English": "Hello there",
}

WARNING_TEXT = {
    "English": """
    **Warning!**
    
    If you are clicking checkbox bellow positive """ + r"$\alpha$" + """ values for CAIF sampling become available.
    It means that language model will be forced to produce toxic or/and abusive text.
    This space is only a demonstration of our method for controllable text generation 
    and we are not responsible for the content produced by this method.
    
    **Please use it carefully and with positive intentions!**
    """,
}


def main():
    st.header("CAIF")
    with open("entropy_cdf.pkl", "rb") as inp:
        x_s, y_s = pickle.load(inp)
    scatter = go.Scatter({
        "x": x_s,
        "y": y_s,
        "name": "GPT2",
        "mode": "lines",
    }
    )
    layout = go.Layout({
        "yaxis": {
            "title": "Speedup",
            "tickvals": [0, 0.5, 0.8, 1],
            "ticktext": ["1x", "2x", "5x", "10x"]
        },
        "xaxis": {"title": "Entropy threshold"},
        "template": "plotly_white",
    })

    language = "English"
    cls_model_name = st.selectbox(
        ATTRIBUTE_MODEL_LABEL[language],
        ATTRIBUTE_MODELS[language]

    )
    lm_model_name = st.selectbox(
        LM_LABEL[language],
        LANGUAGE_MODELS[language]
    )
    cls_model_config = AutoConfig.from_pretrained(cls_model_name)
    if cls_model_config.problem_type == "multi_label_classification":
        label2id = cls_model_config.label2id
        label_key = st.selectbox(ATTRIBUTE_LABEL[language], label2id.keys())
        target_label_id = label2id[label_key]
        act_type = "sigmoid"
    elif cls_model_config.problem_type == "single_label_classification":
        label2id = cls_model_config.label2id
        label_key = st.selectbox(ATTRIBUTE_LABEL[language], [list(label2id.keys())[-1]])
        target_label_id = 1
        act_type = "sigmoid"
    else:
        label_key = st.selectbox(ATTRIBUTE_LABEL[language], ["Negative"])
        target_label_id = 0
        act_type = "softmax"

    st.markdown(r"""In our method, we reweight the probability of the next token with the external classifier, namely, the Attribute model. If $\alpha$ parameter is equal to zero we can see that the distribution below collapses into a simple language model without any modification. If alpha is below zero then every generation step attribute model tries to minimize the probability of the desired attribute. Otherwise, the model is forced to produce text with a higher probability of the attribute.""")
    st.latex(r"p(x_i|x_{<i}, c) \propto p(x_i|x_{<i})p(c|x_{\leq i})^{\alpha}")
    st.write(WARNING_TEXT[language])
    show_pos_alpha = st.checkbox("Show positive alphas", value=False)
    if act_type == "softmax":
        alpha = st.slider("α", min_value=-30, max_value=30 if show_pos_alpha else 0, step=1, value=0)
    else:
        alpha = st.slider("α", min_value=-5, max_value=5 if show_pos_alpha else 0, step=1, value=0)
    with st.expander("Advanced settings"):
        entropy_threshold = st.slider("Entropy threshold", min_value=0., max_value=10., step=.1, value=2.)
        plot_idx = np.argmin(np.abs(entropy_threshold - x_s))
        scatter_tip = go.Scatter({
            "x": [x_s[plot_idx]],
            "y": [y_s[plot_idx]],
            "mode": "markers"
        })
        scatter_tip_lines = go.Scatter({
            "x": [0, x_s[plot_idx]],
            "y": [y_s[plot_idx]] * 2,
            "mode": "lines",
            "line": {
                "color": "grey",
                "dash": "dash"
            }
        })
        figure = go.Figure(data=[scatter, scatter_tip, scatter_tip_lines], layout=layout)
        figure.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor='#FFFFFF', showlegend=False)
        st.plotly_chart(figure, use_container_width=True)
        st.subheader("What is it?")
        st.write("Text generation with an external classifier requires a huge amount of computation. "
                 "Therefore text generating with CAIF could be slow. "
                 "To overcome this issue, we can apply reweighting not for every step. "
                 "Our hypothesis is that we can run reweighting only "
                 "if entropy of the next token is above certain threshold. "
                 "This strategy will reduce the amount of computation. "
                 "Note that if entropy threshold is too high, we don't get desired attribute in generated text")
        fp16 = st.checkbox("FP16", value=True)
        st.write("FP16 computation is faster in comparison with full precision, "
                 "but sometimes could yield Nones (especially with large alphas).")
    st.session_state["generated_text"] = None
    if "sst" in cls_model_name:
        prompt = st.text_input(TEXT_PROMPT_LABEL[language], "The movie")
    else:
        prompt = st.text_input(TEXT_PROMPT_LABEL[language], PROMPT_EXAMPLE[language])
    num_tokens = st.slider("# tokens to be generated", min_value=5, max_value=40, step=1, value=20)
    num_tokens = int(num_tokens)
    st.subheader("Generated text:")

    def generate():
        text = inference(
            lm_model_name=lm_model_name,
            cls_model_name=cls_model_name,
            prompt=prompt,
            alpha=alpha,
            target_label_id=target_label_id,
            entropy_threshold=entropy_threshold,
            fp16=fp16,
            act_type=act_type,
            num_tokens=num_tokens
        )

    st.button("Generate new", on_click=generate())

    st.subheader("Citation")
    st.code(CITE)


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda lm_tokenizer: hash(lm_tokenizer.to_str)}, allow_output_mutation=True)
def load_generator(lm_model_name: str) -> Generator:
    with st.spinner('Loading language model...'):
        generator = Generator(lm_model_name=lm_model_name, device=device)
        return generator


# @st.cache(hash_funcs={tokenizers.Tokenizer: lambda lm_tokenizer: hash(lm_tokenizer.to_str)}, allow_output_mutation=True)
def load_sampler(cls_model_name, lm_tokenizer):
    with st.spinner('Loading classifier model...'):
        sampler = CAIFSampler(classifier_name=cls_model_name, lm_tokenizer=lm_tokenizer, device=device)
        return sampler


def inference(
        lm_model_name: str,
        cls_model_name: str,
        prompt: str,
        fp16: bool = True,
        alpha: float = 5,
        target_label_id: int = 0,
        entropy_threshold: float = 0,
        act_type: str = "sigmoid",
        num_tokens=10,
) -> str:
    torch.set_grad_enabled(False)
    generator = load_generator(lm_model_name=lm_model_name)
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(lm_model_name)
    if alpha != 0:
        caif_sampler = load_sampler(cls_model_name=cls_model_name, lm_tokenizer=lm_tokenizer)
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
        print(f"Generating for prompt: {prompt}")
        progress_bar = st.progress(0)
        sequences, tokens = generator.sample_sequences(
            num_samples=1,
            input_prompt=prompt,
            max_length=num_tokens,
            caif_period=1,
            entropy=entropy_threshold,
            progress_bar=progress_bar,
            **kwargs
        )
        print(f"Output for prompt: {sequences}")

    return sequences[0]


if __name__ == "__main__":
    main()
