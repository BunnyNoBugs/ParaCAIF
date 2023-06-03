Folder `caif` contains the implementation of CAIF from the [huggingface demo](https://huggingface.co/spaces/tinkoff-ai/caif) with some changes and bug fixes we make to enable it to work with paraphraser models:
 * we add support for encoder-decoder models like T5
 * we add support for decoder-based paraphraser models.

The notebooks in this folder contain the inference for plain and ParaCAIF versions of paraphraser models: ruGPT-3 and mT5 from [Russian Paraphrasers](https://github.com/RussianNLP/russian_paraphrasers) library, ruT5 from [this](https://habr.com/ru/articles/564916/) blog post, and T5 baseline from paper "[Text Detoxification using Large Pre-trained Neural Models](https://github.com/s-nlp/detox)".
The contents of the files are the following:
* `caif_russian_paraphrasers.ipynb` -- inference of ParaCAIF Russian Paraphrasers
* `eng_caif_t5.ipynb` -- inference of T5
* `rank_paraphrases.py` -- implementation of candidates reranking
* `rank_samples.ipynb` -- reranking of candidates for models which do not do it during inference
* `russian_paraphrases.ipynb` -- inference of plain Russian Paraphrasers
* `russian_paraphrasers_inference.py` -- inference functions for Russian Paraphrasers
* `rut5_paraphrase_baseline.ipynb` -- inference of ruT5
* `t5_paraphraser_inference.py` -- inference functions for t5
* `utils.py` -- function for output post-processing
