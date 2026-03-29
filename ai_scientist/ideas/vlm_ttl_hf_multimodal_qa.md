# Title: Test-Time Learning for VLMs on Hugging Face Multimodal QA Tasks

## Keywords
vision language models, test-time learning, multimodal adaptation, LoRA, Hugging Face, ScienceQA, ChartQA, DocVQA, TextVQA

## TL;DR
Study whether lightweight test-time adaptation improves open vision-language models on directly downloadable Hugging Face multimodal QA tasks, while keeping the setup reproducible on a single 96GB GPU.

## Abstract
We want to study test-time learning for open vision-language models on multimodal reasoning and question-answering tasks that can be downloaded directly from Hugging Face. The main setting is lightweight adaptation at test time using only the observed input example or unlabeled test stream, without access to gold labels. We are interested in stable and cheap adaptation mechanisms that could work for VLMs, including last-layer feature adaptation, LoRA on selected multimodal layers, and simple self-supervised objectives such as input reconstruction or answer likelihood proxies. Candidate datasets should be directly accessible from Hugging Face, with preference for `lmms-lab/ScienceQA`, `HuggingFaceM4/ChartQA`, `facebook/textvqa`, and `pixparse/docvqa-single-page-questions`. Candidate models should be open and practical for a single 96GB GPU, with preference for VLM families in the 7B to 8B range that support standard Hugging Face loading. The primary questions are: which multimodal tasks benefit from lightweight test-time updates, which parts of the model are safest to adapt, and how adaptation changes both accuracy and runtime cost. The project should prefer strong and simple baselines, careful ablations over adaptation location and step count, and explicit reporting of failure modes such as overfitting to OCR noise or instability on chart and document tasks.
