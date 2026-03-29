# API Keys And Launch

This repo now supports two local secret-loading paths:

- macOS keychain
- local env files such as `.env.local`

The default launch scripts load both automatically.

## Recommended Default Stack

For the current default configs in this repo, the minimum useful key set is:

- `OPENAI_API_KEY`

Optional but useful:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION_NAME`
- `S2_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `HUGGINGFACE_API_KEY`
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

Reason:

- ideation defaults to `gpt-5.4`
- the current BFTS configs now default to `gpt-5.4` for experiment/code generation
- writeup and review defaults also use `gpt-5.4`

## One-Time Setup On macOS

Interactive bootstrap:

```bash
bash scripts/bootstrap_api_keys_macos.sh
```

Store a single key:

```bash
bash scripts/set_api_key_macos.sh OPENAI_API_KEY
```

## File-Based Setup

If you prefer a file instead of keychain:

```bash
cp .env.local.example .env.local
```

Then fill in `.env.local`.

## Check Current Status

Default required keys:

```bash
bash scripts/check_api_env.sh default
```

Show all supported key slots:

```bash
bash scripts/check_api_env.sh all
```

## One-Click Launch

Run ideation only:

```bash
bash scripts/run_ideation.sh
```

Run the scientist pipeline from an ideas JSON:

```bash
bash scripts/run_scientist.sh
```

Run ideation and scientist end-to-end with defaults:

```bash
bash scripts/run_open_pipeline.sh
```

## Current Defaults

- topic file: `ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.md`
- ideation model: `gpt-5.4`
- pipeline config: `configs/bfts_llm_ttl_96gb.yaml`
- first idea index: `0`

These defaults are conservative and meant to work as a stable starting point on a single `96GB` GPU server.
