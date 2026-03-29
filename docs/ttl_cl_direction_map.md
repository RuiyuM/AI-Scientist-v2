# TTL / CL Direction Map

This repo is being customized for open `LLM` and `VLM` research with two hard constraints:

- tasks and datasets should be directly downloadable from Hugging Face
- experiments should be practical on a rented server with a single `RTX Pro 6000 96GB`

The reference repo `subset-selection-core` is only being used as a model for git persistence and server workflow hygiene. It does not constrain the scientific topic here.

## Recommended Research Order

1. `LLM test-time learning`
2. `VLM test-time learning`
3. `continual learning with shared or constrained LoRA updates`

This order is intentional. `LLM TTL` is cheaper, easier to evaluate, and more likely to produce a clean first paper than `VLM TTL` or full continual-learning pipelines.

## Anchor Papers

- `2602.09719`: dynamic layer-wise unsupervised TTA for LLMs via LoRA and a hypernetwork
- `2505.12392`: SLOT, a sample-specific vector added near the output layer
- `2505.20633`: TLM, input-perplexity minimization with selective high-perplexity updates
- `2602.06043`: Share, a shared LoRA subspace for continual adaptation

## Best First Direction

The strongest first direction for this repo is:

`unlabeled test-time learning for open LLMs on Hugging Face QA tasks under domain shift`

Recommended datasets:

- `cais/mmlu`
- `allenai/ai2_arc`
- `qiaojin/PubMedQA`
- `openai/gsm8k`

Recommended model scale for first runs:

- start with open `7B` to `14B` text models
- keep first runs simple enough that one model instance fits comfortably and can be reloaded repeatedly without server instability
- if quantized larger models are used, treat them as follow-up runs rather than first-wave baselines

## Best Second Direction

Once the LLM path is stable, move to:

`lightweight test-time learning for open VLMs on directly downloadable multimodal QA tasks`

Recommended datasets:

- `lmms-lab/ScienceQA`
- `HuggingFaceM4/ChartQA`
- `facebook/textvqa`
- `pixparse/docvqa-single-page-questions`

Recommended model scale for first VLM runs:

- prefer `7B` to `8B` Hugging Face-native VLMs
- do not start with the heaviest available VLM
- keep adaptation local and lightweight

## Continual Learning Scope

The continual-learning direction should be narrower than the test-time-learning direction:

- sequentially adapt across task families instead of trying many unrelated tasks
- keep one shared low-rank parameterization
- explicitly measure forward transfer and forgetting

Suggested sequence:

- `ai2_arc -> PubMedQA -> selected MMLU subjects`

or

- a curated sequence of `MMLU` subject groups

## 96GB GPU Guidance

The `96GB` GPU is large enough to make this repo useful for open-model experimentation, but it is still easy to waste time and money by over-parallelizing.

- keep `num_workers: 1` for first-wave runs
- prefer one stable experiment process over many concurrent OOM-prone processes
- use the lighter `LLM TTL` config before the `VLM TTL` config
- keep git small and push code or small summaries frequently

## Repo Workflow

- topic files live under `ai_scientist/ideas/`
- reusable BFTS presets live under `configs/`
- server autosave scripts live under `scripts/`

Suggested branch layout:

- `main`: synced baseline
- `llm-ttl`: first active paper direction
- `vlm-ttl`: multimodal direction
- `cl-share`: continual-learning direction
