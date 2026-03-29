## Name

cross_modal_consistency_ttl

## Title

Cross-Modal Consistency Signals for Stable Test-Time Learning in Open Vision-Language Models

## Short Hypothesis

For open vision-language models, self-supervised test-time learning based on cross-modal consistency between answer logits, generated rationales, and image-grounded captions should be more stable than plain entropy minimization on multimodal Hugging Face benchmarks.

## Related Work

Most test-time adaptation methods for large models rely on entropy-style objectives or pseudo-label confidence, which can be brittle for multimodal reasoning because the model may be confident for the wrong reason. Vision-language models also expose more latent structure: caption tokens, rationale text, and answer distributions can agree or conflict. This proposal differs by turning that agreement into a test-time learning signal and directly comparing it with simpler entropy-based adaptation on realistic Hugging Face multimodal tasks. The emphasis is on stability, not merely raw average gain, and on understanding whether cross-modal agreement can identify safe opportunities for adaptation.

## Abstract

Test-time learning for vision-language models is appealing but unstable: multimodal models can become overconfident, and naive entropy minimization may reinforce spurious image-text shortcuts. We propose a targeted study of cross-modal consistency signals for stable online adaptation in open VLMs. The idea is to update a lightweight adapter only when the model's answer distribution is supported by consistent auxiliary evidence such as generated captions, rationales, or repeated answer samples. This creates a stronger self-supervised signal than entropy alone and may reduce harmful updates on ambiguous inputs. We will evaluate open models such as Qwen2-VL on Hugging Face multimodal reasoning tasks including `lmms-lab/ScienceQA`, `HuggingFaceM4/ChartQA`, and document-style VQA data when feasible on a single 96GB GPU. Baselines include frozen inference, entropy-minimization adaptation, and confidence-gated LoRA adaptation without consistency checks. We will measure answer accuracy, stability over the stream, adaptation frequency, latency, and failure modes tied to OCR, chart parsing, or distractor-heavy images. The expected contribution is a concrete answer to whether cross-modal agreement provides a safer trigger and objective for multimodal test-time learning.

## Experiments

- Build a VLM-only evaluation path around an open model such as Qwen2-VL-7B-Instruct and Hugging Face multimodal tasks including `lmms-lab/ScienceQA` and `HuggingFaceM4/ChartQA`, adding document-style VQA only if stable.
- Implement four baselines: frozen inference, entropy-minimization test-time adaptation, confidence-gated LoRA adaptation, and consistency-gated adaptation using agreement between answer logits, rationale generations, and caption-like auxiliary text.
- Compare adaptation on easy versus hard subsets defined by OCR load, visual clutter, or long-context reasoning needs.
- Track exact-match or multiple-choice accuracy, trigger rate, latency, and instability events where performance degrades after adaptation.
- Study whether consistency signals are most useful on chart and document problems where a single view of the image is not enough.
- Produce qualitative examples showing when consistency gating blocks harmful updates and when it still fails.

## Risk Factors And Limitations

- Generating auxiliary rationales or captions may add too much latency for a practical test-time learning loop.
- Agreement between weak signals may still be wrong, so consistency is not a guarantee of correctness.
- Chart and OCR-heavy benchmarks may fail because of perception bottlenecks rather than adaptation policy.
- This idea may be engineering-heavy and could need simplification before a full paper-quality run.

