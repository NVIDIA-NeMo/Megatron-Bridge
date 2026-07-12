# Full-model validation inventory

Last updated: 2026-07-12

This manifest tracks full-checkpoint conversion and inference validation for the
model architectures and explicitly named variants in the root
[Supported Models](../../README.md#supported-models) table. It selects one
canonical public, complete Hugging Face checkpoint per architecture/version;
it does not try to enumerate every size or instruction-tuned derivative. Sizes
explicitly named in the root table are tracked separately.

`Params` is total architecture parameters, not parameters activated per token.
For packed low-precision checkpoints, it follows the model architecture/model
card rather than the raw stored tensor-element count. Revisions are immutable
Hugging Face commit SHAs resolved on the date above. Status is one of `TODO`,
`DOWNLOADING`, `RUNNING`, `FIXING`, `PASS`, or `BLOCKED`.

| Family | Architecture / version | Canonical HF checkpoint | Revision | Params | Modality | Example path | Status |
|---|---|---|---|---:|---|---|---|
| Bailing | Ling 2.0 (Flash) | `inclusionAI/Ling-flash-2.0` | `18ca64a019b553be57bab50af3207fb2f3675edc` | 100B | text | `examples/models/bailing` | TODO |
| Bailing | Ling MoE V2 / Bailing (Mini) | `inclusionAI/Ling-mini-2.0` | `920c3fd9916e3d5e543fc4f609e827cad8a32983` | 16B | text | `examples/models/bailing` | TODO |
| DeepSeek | DeepSeek V2 | `deepseek-ai/DeepSeek-V2` | `4461458f186c35188585855f28f77af5661ad489` | 235.7B | text | `examples/conversion` | TODO |
| DeepSeek | DeepSeek V2 Lite | `deepseek-ai/DeepSeek-V2-Lite` | `604d5664dddd88a0433dbae533b7fe9472482de0` | 15.7B | text | `examples/conversion` | TODO |
| DeepSeek | DeepSeek V4 Flash | `deepseek-ai/DeepSeek-V4-Flash-Base` | `8855555deef230a27a21a8d6f294b7b7497759b6` | 292.0B | text | `examples/models/deepseek_v4` | TODO |
| Diffusion | FLUX.1 | `black-forest-labs/FLUX.1-dev` | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` | 12B | text-to-image | `examples/models/flux` | TODO |
| Diffusion | LLaDA 1.5 | `GSAI-ML/LLaDA-1.5` | `84346fd91ba60252d260022201ad6fc5a3468fb2` | 8.0B | text diffusion | `examples/models/llada/llada15` | TODO |
| Diffusion | Nemotron-Labs Diffusion | `nvidia/Nemotron-Labs-Diffusion-3B` | `0d51902da1f8869f83413ce642fab402fa5641e0` | 3.8B | text diffusion | `examples/models/nemotron_labs_diffusion` | TODO |
| Diffusion | WAN 2.1 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `0fad780a534b6463e45facd96134c9f345acfa5b` | 1.3B | text-to-video | `examples/models/wan` | TODO |
| Ernie | Ernie 4.5 MoE | `baidu/ERNIE-4.5-21B-A3B-PT` | `87db95487941cb39592ee0abca3b9155a6d19c5c` | 21.9B | text | `examples/conversion` | TODO |
| Ernie | Ernie 4.5 VL MoE | `baidu/ERNIE-4.5-VL-28B-A3B-PT` | `e3815e65c607ea211bfe21b46ab0cd264b76731c` | 29.4B | vision-language | `examples/models/vlm/ernie_vl` | TODO |
| Falcon | Falcon H1 | `tiiuae/Falcon-H1-0.5B-Instruct` | `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368` | 0.5B | text | `examples/models/falcon_h1` | TODO |
| Gemma | Gemma | `google/gemma-2b` | `9cf48e52b224239de00d483ec8eb84fb8d0f3a3a` | 2.5B | text | `examples/conversion` | TODO |
| Gemma | Gemma 2 | `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | 2.6B | text | `examples/conversion` | TODO |
| Gemma | Gemma 3 | `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | 1.0B | text | `examples/conversion` | TODO |
| Gemma | Gemma 3-VL | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | 4.3B | vision-language | `examples/models/gemma/gemma3_vl` | TODO |
| Gemma | Gemma 4 26B-A4B MoE | `google/gemma-4-26B-A4B` | `6b556d30bb65a6ee0bdaec99bab0afc7bf1494fb` | 26.5B | text | `examples/models/gemma/gemma4` | TODO |
| Gemma | Gemma 4 31B dense | `google/gemma-4-31B` | `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3` | 32.7B | text | `examples/models/gemma/gemma4` | TODO |
| Gemma | Gemma 4-VL 26B-A4B MoE | `google/gemma-4-26B-A4B-it` | `5305c1e72ea29c01f31a81230d52b375ba88b409` | 26.5B | vision-language | `examples/models/gemma/gemma4_vl` | TODO |
| GLM | GLM-4.5 | `zai-org/GLM-4.5` | `cbb2c7cfb52fa128a9660cb1a7a78e017899e115` | 358.3B | text | `examples/conversion` | TODO |
| GLM | GLM-4.7 | `zai-org/GLM-4.7` | `602d01efcdd332c5238ca4bcede555defbe83eb7` | 358.3B | text | `examples/models/glm47` | TODO |
| GLM | GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 31.2B | text | `examples/models/glm47` | TODO |
| GLM | GLM-4.5V | `zai-org/GLM-4.5V` | `ed47433b37111465ec527affaaddceff371bca04` | 107.7B | vision-language | `examples/models/glm/glm_45v` | TODO |
| GPT-OSS | GPT-OSS 20B | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | 21.5B | text | `examples/models/gpt_oss` | TODO |
| GPT-OSS | GPT-OSS 120B | `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | 120.4B | text | `examples/models/gpt_oss` | TODO |
| HY V3 | Hy3 preview-Base | `tencent/Hy3-preview-Base` | `54a62bb00a50195423bffb6b55e91aa28b6a8ce2` | 298.8B | text | `examples/conversion` | TODO |
| Llama | Llama 2 | `meta-llama/Llama-2-7b-hf` | `01c7f73d771dfac7d292323805ebc428287df4f9` | 6.7B | text | `examples/conversion` | PASS |
| Llama | Llama 3 | `meta-llama/Meta-Llama-3-8B` | `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920` | 8.0B | text | `examples/conversion` | TODO |
| Llama | Llama 3.1 | `meta-llama/Meta-Llama-3.1-8B` | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | 8.0B | text | `examples/conversion` | TODO |
| Llama | Llama 3.2 | `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | 1.2B | text | `examples/conversion` | TODO |
| Llama | Llama 3.3 | `meta-llama/Llama-3.3-70B-Instruct` | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | 70.6B | text | `examples/conversion` | TODO |
| MiniMax | MiniMax-M2 | `MiniMaxAI/MiniMax-M2` | `757303d492a50514c312788b5247a4f696a4c6a3` | 456B | text | `examples/models/minimax/minimax_m2` | TODO |
| MiniMax | MiniMax-M2.5 | `MiniMaxAI/MiniMax-M2.5` | `f710177d938eff80b684d42c5aa84b382612f21f` | 456B | text | `examples/models/minimax/minimax_m2` | TODO |
| MiniMax | MiniMax-M2.7 | `MiniMaxAI/MiniMax-M2.7` | `d494266a4affc0d2995ba1fa35c8481cbd84294b` | 456B | text | `examples/models/minimax/minimax_m2` | TODO |
| Mistral | Mistral | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | 7.2B | text | `examples/conversion` | TODO |
| Mistral | Ministral 3 3B | `mistralai/Ministral-3-3B-Base-2512` | `6f9c4b12a95b139af68670a6713616b757923735` | 4.3B | vision-language | `examples/models/mistral/ministral3` | TODO |
| Mistral | Ministral 3 8B | `mistralai/Ministral-3-8B-Base-2512` | `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef` | 8.9B | vision-language | `examples/models/mistral/ministral3` | TODO |
| Mistral | Ministral 3 14B | `mistralai/Ministral-3-14B-Base-2512` | `5b0ceedbb42dff466ae60b258ba296f32da51384` | 13.9B | vision-language | `examples/models/mistral/ministral3` | TODO |
| Xiaomi-MiMo | MiMo | `XiaomiMiMo/MiMo-7B-Base` | `c72df4586cb8bdeebd65f36929cd3385a6566fbe` | 7.8B | text | `examples/megatron_mimo` | TODO |
| Xiaomi-MiMo | MiMo-V2-Flash | `XiaomiMiMo/MiMo-V2-Flash` | `1afd314a2406c282e0956375c34a676501c78649` | 309.8B | text | `examples/models/mimo_v2_flash` | TODO |
| Moonlight | Moonlight | `moonshotai/Moonlight-16B-A3B` | `476b36a473d4467f94469414bef6cee75c9c8172` | 16.0B | text | `examples/conversion` | TODO |
| Nemotron | Nemotron H | `nvidia/Nemotron-H-4B-Base-8K` | `faba3b731ad7ea5781b9518ae75fb610a94affcf` | 4.5B | text | `examples/conversion` | TODO |
| Nemotron | Nemotron Nano v2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` | `dc0661c829b14e5b9246c05cfa89094a0875e052` | 8.9B | text | `examples/conversion` | TODO |
| Nemotron | Nemotron-3 Nano | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | `97ab8012882a655dc38df4fee47422aca9caca07` | 31.6B | text | `examples/models/nemotron/nemotron_3/nano` | TODO |
| Nemotron | Nemotron-3 Super | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | `d51eab0d1f979ebc26b546e634a04f450d99158e` | 123.6B | text | `examples/models/nemotron/nemotron_3/super` | TODO |
| Nemotron | Llama Nemotron | `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | `d552708a9d575fa8d4a690b988fd870d65279f98` | 4.5B | text | `examples/conversion` | TODO |
| Nemotron | Nemotron Nano v2 VL | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | `5d250e2e111dc5e1434131bdf3d590c27a878ade` | 13.2B | vision-language | `examples/models/nemotron/nemotron_vl` | TODO |
| Nemotron | Nemotron-3 Nano Omni | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | `24e67ea000b7c2837fc8f9488aa2008524fac8ba` | 33.0B | vision-language-audio | `examples/models/nemotron/nemotron_3_omni` | TODO |
| OLMoE | OLMoE | `allenai/OLMoE-1B-7B-0125` | `9b0c1aa87e34a20052389dce1f0cf01da783f654` | 6.9B | text | `examples/conversion` | TODO |
| Qwen | Qwen2 | `Qwen/Qwen2-7B` | `453ed1575b739b5b03ce3758b23befdb0967f40e` | 7.6B | text | `examples/conversion` | TODO |
| Qwen | Qwen2.5 | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` | 7.6B | text | `examples/conversion` | TODO |
| Qwen | Qwen3 | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | 8.2B | text | `examples/conversion` | TODO |
| Qwen | Qwen3-MoE | `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 30.5B | text | `examples/conversion` | TODO |
| Qwen | Qwen3 Next | `Qwen/Qwen3-Next-80B-A3B-Instruct` | `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` | 81.3B | text | `examples/models/qwen/qwen3_next` | TODO |
| Qwen | Qwen3.5 dense | `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | 27.8B | text | `examples/models/qwen/qwen35_vl` | TODO |
| Qwen | Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B` | `59d61f3ce65a6d9863b86d2e96597125219dc754` | 36.0B | text | `examples/models/qwen/qwen35_vl` | TODO |
| Qwen | Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` | 3.8B | vision-language | `examples/models/qwen/qwen_vl` | TODO |
| Qwen | Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 8.8B | vision-language | `examples/models/qwen/qwen3_vl` | TODO |
| Qwen | Qwen3.5-VL | `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | 27.8B | vision-language | `examples/models/qwen/qwen35_vl` | TODO |
| Qwen | Qwen3.6-VL | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 36.0B | vision-language | `examples/models/qwen/qwen35_vl` | TODO |
| Qwen | Qwen2 Audio | `Qwen/Qwen2-Audio-7B-Instruct` | `0a095220c30b7b31434169c3086508ef3ea5bf0a` | 8.4B | audio-text | `examples/models/qwen/qwen2_audio` | TODO |
| Qwen | Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-7B` | `ae9e1690543ffd5c0221dc27f79834d0294cba00` | 10.7B | vision-audio-text | `examples/models/qwen/qwen25_omni` | TODO |
| Qwen | Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `26291f793822fb6be9555850f06dfe95f2d7e695` | 35.3B | vision-audio-text | `examples/models/qwen/qwen3_omni` | TODO |
| Qwen | Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` | 2.3B | audio-text | `examples/models/qwen/qwen3_asr` | PASS |
| Sarvam | Sarvam | `sarvamai/sarvam-30b` | `071ae95e933605ca1104a6b4524a36a98488efa4` | 32.2B | text | `examples/models/sarvam` | TODO |
| StepFun | Step-3.5-Flash | `stepfun-ai/Step-3.5-Flash` | `ab446a3de5e171ea341227e24bb1f090e1b771f7` | 199.4B | text | `examples/models/stepfun/step35` | TODO |
| StepFun | Step-3.7-Flash | `stepfun-ai/Step-3.7-Flash` | `5f6244077ac62e04eec3f320501ff8c2b293373a` | 201.4B | vision-language | `examples/models/stepfun/step37` | TODO |

## Excluded at 500B or above

These root-table architectures have public checkpoints but are outside this
program because their total parameter count is at least 500B. Active parameter
counts are deliberately not used for this decision.

| Architecture / version | Evidence checkpoint | Revision | Total params | Evidence |
|---|---|---|---:|---|
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3-Base` | `afb92e1fa402c2be2a9eb085312bb02e0384d6c7` | 684.5B | Complete checkpoint tensor inventory; model card reports 671B nominal. |
| DeepSeek V4 / Pro | `deepseek-ai/DeepSeek-V4-Pro-Base` | `98730c030fbdbaca4950788280a35c4642b208a9` | 1.60T | Complete checkpoint tensor inventory. |
| GLM-5 | `zai-org/GLM-5` | `4e6698ba8e85059d749020e3c4d2123719f23926` | 753.9B | Complete checkpoint tensor inventory. |
| GLM-5.1 | `zai-org/GLM-5.1` | `26e1bd6e011feb778d25ae34b09b07074139d92d` | 753.9B | Complete checkpoint tensor inventory. |
| Kimi K2 | `moonshotai/Kimi-K2-Instruct` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | 1.03T | Complete checkpoint tensor inventory. |
| Kimi-K2.5-VL | `moonshotai/Kimi-K2.5` | `4d01dfe0332d63057c186e0b262165819efb6611` | 1.06T | Complete checkpoint tensor inventory. |

## PASS acceptance

A row moves to `PASS` only after validation uses the complete checkpoint and
records, as applicable: original-framework deterministic inference, full
HF-to-Megatron import, Megatron inference, full Megatron-to-HF export,
exported-HF inference, exact state-dict round-trip for pure non-quantized
mappings, and BF16 logit cosine above 0.9999. For quantized sources the report
must instead state source/export dtype, dequantization behavior, aggregate
weight error, and logit or deterministic token parity; a lossy conversion is
never described as exact.
