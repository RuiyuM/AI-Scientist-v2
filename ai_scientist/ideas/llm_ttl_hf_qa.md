# Title: Test-Time Learning for Domain-Shifted LLMs on Hugging Face QA Tasks

## Keywords
large language models, test-time learning, test-time adaptation, LoRA, domain shift, Hugging Face, question answering

## TL;DR
Study whether lightweight unlabeled test-time adaptation improves open LLMs on Hugging Face QA tasks under domain shift, while keeping compute and implementation simple enough for one rented GPU server.

## Abstract
We want to study test-time learning for open large language models under realistic distribution shift using tasks that can be downloaded directly from Hugging Face. The main setting is unlabeled test-time adaptation: the model may adapt at inference time using only the input prompt or the test stream, without access to gold answers. We are especially interested in lightweight and stable methods inspired by recent work on sample-specific optimization, input-perplexity minimization, and LoRA-based adaptation. Candidate tasks should come from Hugging Face datasets such as `cais/mmlu`, `allenai/ai2_arc`, `qiaojin/PubMedQA`, and `openai/gsm8k`, with preference for tasks that are easy to evaluate and easy to reproduce on a single server. Candidate models should be open and practical for a 96GB GPU, with preference for model families such as Qwen and Llama in the 7B to 14B range, or larger models in quantized form when stable. The main research questions are: when does unlabeled test-time adaptation help, which lightweight parameterization is most stable, and how much extra compute per sample is justified by the gain in accuracy. Experiments should compare no adaptation against a small number of simple adaptation variants, including last-layer sample-specific vectors, LoRA on selected transformer blocks, and selective updates driven by high input perplexity. The paper should emphasize strong baselines, careful failure analysis, compute overhead, and negative results when adaptation hurts.
