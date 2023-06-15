# ParaCAIF

## Description

This repository contains code and data for a work submitted as a bachelor thesis in HSE University.
In this work, we adapt [CAIF](https://huggingface.co/spaces/tinkoff-ai/caif), an existing controllable text generation (CTG) method, for text style transfer (TST).

To adapt CAIF for TST, we replace a regular language model (LM) with a paraphraser and add reranking according to style transfer accuracy and semantic similarity.
To illustrate the applicability of the resulting method, ParaCAIF, we apply it to a rather new yet practical subtask of TST – detoxification.
We conduct the experiments with detoxification for two languages – Russian and English.
For Russian, we evaluate our models on the data of [RUSSE-2022 Detoxification competition](https://github.com/s-nlp/russe_detox_2022).
For English, we work with the test data used in an EMNLP 2021 paper "[Text Detoxification using Large Pre-trained Neural Models](https://github.com/s-nlp/detox)".

The slides for the bachelor thesis presentation can be found [here](https://docs.google.com/presentation/d/1S8KD3EIXQIJF3Xt1t4UyI81YZ311Gyn22tPgWd8GSSM/edit?usp=sharing).
The bibliography used in the work can be found in a [Zotero group](https://www.zotero.org/groups/4893101/controllable_text_generation/library).

## Repository structure

Folder `data` contains the inputs and outputs of the models for detoxification in Russian and English, as well as evaluation results for English.

Folder `evaluation` contains the evalution results for Russian, as well as their analysis.

Folder `paracaif` contains the implementation of ParaCAIF and notebooks with inference for detoxification experiments.

Folder `toxicity_classification` contains training notebooks for Russian toxicity classifiers which are used as attribute discriminators in ParaCAIF.
