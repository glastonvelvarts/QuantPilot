# NVIDIA test session: RTX 4060 Laptop, Windows 11 + WSL2

Results of the Phase 4 test plan on an NVIDIA PC, for the Mac side to pick up. Everything here was run on
branch `nvidia-windows-test` (= `phase-4-runtimes` + the fixes listed below), 2026-11-08.

## The machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8,188 MiB, compute capability 8.9, driver 621.01 (WDDM) |
| Free VRAM | 7,520–7,640 MiB with the desktop running (Cursor, Explorer, Brave hold the rest) |
| CPU / RAM | Ryzen 7 7840HS, 23.2 GB (10–14 GB free during the runs) |
| llama.cpp | b11429, `win-cuda` (CUDA 12.4) build that QuantPilot downloads |
| vLLM | 0.33.0 in WSL2 Ubuntu (torch 2.14.0+cu130), separate venv `~/.venvs/vllm` |
| Models | the 8 Mac GGUFs, byte for byte (sizes checked against `tests/tests/data/gguf/*.json.gz`), plus `ggml-org/gpt-oss-20b-GGUF` MXFP4 (12.1 GB, does not fit 8 GB) |

How to rerun on any machine: `scripts/gpu_check.py` (verify, llama-server, MoE benchmarks; picks the
llama.cpp build for the OS through `tool_path`), `scripts/vllm_check.sh` (Linux or WSL2),
`scripts/summarize_verify.py`, `scripts/summarize_moe.py`. Logs are gzipped: `zcat verify/x.log.gz`.

## Test plan, item by item

### 1. llama.cpp's working memory on the GPU: measured

`verify/summary.md` (llama-completion, 9 models × 4k/16k/64k/16k-q8_0) and `server/summary.md`
(llama-server, 9 models × 4k/16k). Every log is in `verify/` and `server/`.

- **The CUDA build has a third buffer kind, `CUDA_Host`** (pinned RAM). `parse_log` counted it as GPU,
  so the GPU working memory was overstated and the CPU side always read 0. Fixed here (`in_ram()` in
  `verify.py`). Example, OLMoE 4k: CUDA0 118.27 MiB + CUDA_Host 20.02 MiB, reported before as
  138.29 on the GPU.
- **Pinned-RAM compute buffer (llama-completion) grows by exactly 1 MiB per 1,024 tokens of context,
  for every model**: 12.02 / 24.02 / 72.02 MiB at 4k / 16k / 64k for Falcon-H1, LFM2, Granite 3.1;
  the 4k base differs by model (Granite 4.0-H 10.02, Gemma 3 14.03, Jamba 24.02, Gemma 2 26.03,
  gpt-oss 27.28). Candidate rule to check against the Mac's CPU buffer numbers.
- **llama-server, GPU compute buffer: +12.00 MiB from 4k to 16k for 8 of 9 models** (Gemma 2:
  +24.00), from a per-model base (29.69 MiB Granite 4.0-H … 206.99 MiB gpt-oss). Only two context sizes
  per model, so a hypothesis, not a rule yet.
- **llama-server reserves far less than llama-completion** (what `verify` runs): LFM2 4k 39.01 vs
  140.02 MiB, Gemma 2 4k 80.02 vs 526.02, gpt-oss 4k 206.99 vs 416.01. Predicting from verify's
  numbers would overstate a server by 100–450 MiB.
- Base GPU compute differs a lot by model at 4k (llama-completion): Falcon-H1 76, LFM2 140,
  Granite 4.0-H 206, Gemma 2 526, Gemma 3 526, gpt-oss 416 MiB.

### 2. Is the input embedding table kept in RAM on CUDA: yes

`weights on the CPU (file range)` matches exactly for every model (`CPU_Mapped`), and `weights on the
GPU` matches for every model but the two Gemmas, with tied embeddings (Llama 3.2 1B) and untied
(Qwen2.5 7B) alike. So the VRAM estimate should leave `token_embd` out, as `verify.predict` does.

**Gemma only, CUDA only:** GPU weights are 0.04 MiB (Gemma 3 1B: 651.63 predicted vs 651.67) and
0.02 MiB (Gemma 2 2B: 1,323.24 vs 1,323.26) larger than the sum of the tensors. It is not per-tensor
alignment: padding every tensor to 32–512 bytes still gives 651.63. Gemma 3 fails the 0.03 MiB
tolerance at every context, Gemma 2 just passes. Something llama.cpp adds for Gemma on a separate GPU;
the Mac's file-range layout would not show it.

### 3. Experts in RAM for a MoE: QuantPilot's plan is too tight

`moe/summary.md`, llama-bench `-p 512 -n 128 -r 3`.

gpt-oss-20b MXFP4 (12.1 GB) on 8 GB:

| setup | prompt tok/s | generate tok/s |
|---|---|---|
| `--n-cpu-moe 6` (QuantPilot's plan) | 250.1 ± 1.3 | 17.0 ± 0.1 |
| `--n-cpu-moe 8` | 249.1 ± 2.4 | 18.1 ± 0.2 |
| `--n-cpu-moe 10` | 544.9 ± 11.6 | 21.2 ± 0.3 |
| `--n-cpu-moe 24` (all experts in RAM) | 362.1 ± 4.0 | 15.2 ± 1.8 |
| `-ngl 15` (whole layers, best) | 526.3 ± 4.0 | 21.9 ± 0.3 |
| `-ngl 18` / `-ngl 21` | 222.3 / 248.6 | 15.0 / 14.3 |

- **The plan overfills the card.** With `--n-cpu-moe 6`, llama.cpp puts **8,522.54 MiB** of weights on
  the GPU (verify predicts 8,522.52, so the layout math is right), plus 114 MiB KV and 416 MiB compute,
  on a card with 7,600 MiB free. The planner's fit test disagrees with verify's layout: it should
  have chosen about 10, the first N that fits, and exactly where the speed peaks. Windows hides this:
  the WDDM driver spills to system RAM (sysmem fallback) instead of failing, so it only gets slower; on
  Linux expect an out-of-memory error. The planner also needs the compute buffer in its budget.
- Experts in RAM against whole layers in RAM, each at its best: 21.2 vs 21.9 tok/s generating,
  545 vs 526 prompt. About even on this PC (PCIe 4.0 x8 laptop, DDR5).
- For MoEs that fit, each expert layer moved to RAM costs speed steadily: OLMoE 104.1 → 78.1 / 71.8 /
  58.4 tok/s at 0 / 4 / 8 / 16; Granite 3.1 MoE 144.4 → 123.3 / 100.9 / 88.9 at 0 / 6 / 12 / 24.
- `verify` with QuantPilot's plan matched exactly for gpt-oss at 4k, 16k, 64k and 16k-q8_0 (KV,
  GPU weights, CPU file range), so the expert layout itself is right on CUDA.

### 4. vLLM's memory budget and start-up check: the check is wrong on this card

`vllm-wsl2/`, Qwen/Qwen2.5-1.5B-Instruct-AWQ.

- **vLLM at its defaults never starts here**: `Free memory on device cuda:0 (6.88/8.0 GiB) on startup
  is less than desired GPU memory utilization (0.92, 7.36 GiB)`, three times out of three.
  `nvidia-smi` showed 7,551–7,640 MiB (7.37–7.46 GiB) free just before. **vLLM measures the free
  memory after creating its own CUDA context, about 0.5 GiB later.** QuantPilot said Perfect, with no
  start-up note. The check needs that overhead: 0.92 × 8 GiB can never be met on an 8 GB card.
- **"As stored" rows of quantized repos skip the vLLM rules.** An AWQ repo gets backend `original`,
  runtime transformers, so neither the budget nor the start-up note applies, and at 32k it even
  said `gpu+ram`, which vLLM cannot do. That is the model a user would point vLLM at.
- **Stored but tied `lm_head` counted as runtime memory.** The AWQ repo stores `lm_head` (0.435 GiB)
  although `config.json` ties it; `inspect` notices it. vLLM does not load it: 1.50 − 0.435 =
  1.07 GiB, and vLLM logs `Model loading took 1.1 GiB`.
- With `--gpu-memory-utilization 0.8` (6.4 GiB) it starts. vLLM's breakdown: weights + non-torch
  1.3 GiB, peak activations 0.59–0.79 GiB (differs between two identical starts), CUDA graphs
  0.2 GiB (0.36 estimated), KV cache 4.3–4.5 GiB.
- **KV bytes per token match**: 4.3 GiB → 161,200 tokens and 4.27 GiB → 160,032 tokens = 28,650 B,
  QuantPilot 28,672 B (`inspect`).

### 5. 8-bit KV cache in vLLM (compute capability 8.9): confirmed

At 32k, `--gpu-memory-utilization 0.8`, Triton attention: f16 4.27 GiB → 160,032 tokens, fp8
4.28 GiB → 320,368 tokens = 14,345 B/token, half of f16. QuantPilot's rule (8 bits at 8.9+) holds.

What it took to run vLLM in WSL2 without root (worth a README line): no gcc, so torch.compile fails
("Failed to find C compiler"); `CC` pointed at `zig cc` from the `ziglang` wheel works. No CUDA
toolkit, so FlashInfer cannot JIT: `VLLM_USE_FLASHINFER_SAMPLER=0`, and for an FP8 cache
`--attention-backend TRITON_ATTN` (FlashAttention 2 has no FP8 cache before Hopper, vLLM then picks
FlashInfer). `scripts/vllm_check.sh` sets these.

### 6. Several GPUs: not tested, one card.

## Fixed on this branch

| Problem | Fix |
|---|---|
| `ocq models` labels mixed `\` and `/` on Windows (`a\b/x.gguf`), same for the `~/…` refs | `as_posix()` in `discovery.py` |
| 4 discovery tests used `symlink_to`, which Windows refuses without Developer Mode | copy where links are refused, as huggingface_hub does |
| `ggml-org/gpt-oss-20b-GGUF` read as an 855M BF16 model with an MXFP4 row at 113 bits/weight: its EAGLE3 draft files sort first and their header was read | `model_header()`: a header whose `general.architecture` is `clip` or `eagle3` drops that file name; now 20.91B, cross-checked with the HF API, 4.63 bits |
| `verify` counted `CUDA_Host` (pinned RAM) as GPU memory | `in_ram()` |
| A llama.cpp crash showed as a "differs +486 MiB" table | `parse_log` reports `fatal error` / `GGML_ASSERT`; `--save-log` writes the log before anything can fail |
| `.env` (HF token) not ignored | `.gitignore` |

`make check`: 335 tests pass, ruff clean. `pytest -m slow` (real GGUF, AWQ, GPTQ, BNB, NVFP4
quantization on this GPU): 5 passed.

## Still open, for the Mac side

1. Planner's `--n-cpu-moe` for gpt-oss-20b: picks 6, needs about 10 (section 3); and the llama.cpp
   compute buffer in the VRAM budget.
2. vLLM start-up check: subtract vLLM's own CUDA context (~0.5 GiB measured) from the free VRAM
   (section 4).
3. Apply the vLLM rules to the "as stored" rows of AWQ / GPTQ / NVFP4 / FP8 repos.
4. Leave a stored `lm_head` out of runtime memory when `config.json` ties it.
5. Speed of a dense `gpu+ram` row: Qwen2.5-7B Q4_K_M at 131k, llama.cpp puts 14/29 layers on the
   GPU; QuantPilot shows `<= 45 tok/s`, measured 13.72 ± 0.46 (all 29 layers: 44.03 ± 2.19). The
   bound ignores the layers in RAM. Same for OLMoE at 64k (11/17 layers on the GPU, shown `<= 446`).
   `moe/qwen2.5-7b-dense-offload.md`.
6. QuantPilot does not predict how many layers llama.cpp's fit puts on the GPU, so `verify` skips
   the weight checks then (Qwen 7B 131k, OLMoE 64k).
7. Gemma's extra 0.02–0.05 MiB on CUDA (section 2), in every Gemma GGUF, also the ones QuantPilot
   converts itself (`gemma-e2e.txt`).
13. The context estimate adds an output buffer of 512 rows × vocab × 4 bytes (`EVAL_BATCH` in
    `model_size.py`): 512 MiB for Gemma 3 (262k vocabulary), 500 MiB for Gemma 2. llama.cpp's
    output buffer on CUDA is 1.00 MiB with llama-completion (1 row) and 4.00 MiB with llama-server
    (4 rows). The 512-row size only resembles llama-completion's GPU compute buffer for Gemma
    (526 MiB); llama-server's is 56–80 MiB. So `feasibility` shows Gemma 3's context at 4k as
    0.57 GB where the cache is 71 MiB. Verify's note "llama-server keeps 4 rows of logits instead of
    512" does not hold for llama-completion on this build either: it kept 1.
8. llama.cpp b11429 crashes on Granite 3.1 MoE with a q8_0 cache at 16k (`ggml-impl.h:330: fatal error`
   in the K-rotation graph, `attn_inp_k_rot`); f16 at 4k/16k/64k is fine. A llama.cpp bug, but
   QuantPilot plans q8_0 there.
9. `feasibility` on a GGUF repo says "--n-cpu-moe, see ocq plan", but `plan` refuses GGUF repos
   ("already GGUF"), so the N is never shown.
10. A gated repo (Gemma, Llama) without an accepted license shows "tensor headers not readable",
    architecture unknown; it should say the repo is gated and link the license page (403
    GatedRepoError from the Hub).
11. `ocq models` lists encoders, an ASR model and CTranslate2 Whisper folders as LLMs with tok/s
    (`intfloat/multilingual-e5-small`, `sentence-transformers/*`, `nvidia/nemotron-3.5-asr-…`,
    `Systran/faster-whisper-*` as "only config.json" although complete).
12. One unit test (`test_ocq_models_rates_each_model_as_stored_offline`) failed once while the slow
    tests ran in parallel and passed in every run after.

## Also checked, all fine

- `ocq verify` on CUDA, dense: Llama 3.2 1B Q8_0 (Ollama) at 4k f16 and 16k q8_0, Qwen2.5 7B Q4_K_M
  at 8k: KV, GPU weights and CPU file range exact.
- Mamba / convolution state on CUDA (Granite 4.0-H, Falcon-H1, Jamba, LFM2): exact at every context.
- `ocq optimize Qwen/Qwen2.5-1.5B-Instruct -b gguf --gguf-type Q4_K_M --execute`: 0.918 GB on disk vs
  0.913 GB predicted, validation passed, 98.4 tok/s measured (expected about 125), 84 ms to first token.
- `ocq models` on this PC: Ollama and the Hugging Face cache found, rated offline.
- Gated repos with the license accepted (`google/gemma-3-1b-it`, `google/gemma-2-2b-it`): `inspect`
  reads the safetensors headers, everything cross-checked with the HF API; the KV cache from
  config.json matches llama.cpp (Gemma 3 38 MiB at 4k for 1 conversation, 71 MiB for 4; Gemma 2
  416 MiB). Full `optimize -b gguf --gguf-type Q4_K_M --execute`, then `verify` at 4k and 32k
  (`gemma-e2e.txt`):

  | | Gemma 3 1B | Gemma 2 2B |
  |---|---|---|
  | Q4_K_M on disk vs predicted | 0.751 vs 0.745 GB | 1.591 vs 1.586 GB |
  | validation | pass | pass |
  | tok/s measured vs expected | 108.8 vs about 137 | 75.9 vs about 86 |
  | time to first token | 80 ms | 92 ms |
  | verify: KV / CPU range / GPU weights | exact / exact / −0.05 MiB | exact / exact / −0.03 MiB |

  The expected speeds, fitted to this PC's earlier runs, are 12–25% high on all three models
  quantized this session (Qwen2.5 1.5B: 98.4 vs about 125).

## Mac side, 2026-11-08 (branch `phase-4-nvidia-fixes`)

Fixed: 1 (expert tensors of a GGUF were counted twice: `--n-cpu-moe` 6 → 11 for gpt-oss), 2 (CUDA
context, now for every runtime), 3, 4, 5, 9, 10, 11, 13. llama.cpp on a separate GPU is now
budgeted as stock llama.cpp b11429 does: 0.55 GB CUDA context, 1 GB `--fit-target` headroom per
card (none with `--n-cpu-moe`), input embeddings in RAM, llama-server's buffers (an upper bound
of all 18 runs in `server/`, at most 90 MiB over).

To re-test on the NVIDIA PC: `feasibility` of gpt-oss-20b (expect `--n-cpu-moe 11`, and whether
it runs without spilling), Qwen2.5-7B at 131k (layers on the GPU vs QuantPilot's 15 of 29),
and vLLM at `--gpu-memory-utilization` 0.8 (now rated for vLLM).

Still open: 6 (verify does not predict llama.cpp's own layer split), 7 (Gemma's extra 0.02–0.05
MiB on CUDA), 8 (llama.cpp crash, Granite MoE + q8_0 at 16k), 12 (did not fail in 30 runs under
load on the Mac), and llama-server's working memory for MLA models (DeepSeek reserved 754 MiB at
64k on the Mac, more than the estimate).

## NVIDIA PC re-test of `phase-4-nvidia-fixes`, 2026-11-08

`retest/`. 354 tests pass on Windows, ruff clean.

**gpt-oss-20b: fixed.** `feasibility` gives `--n-cpu-moe 11`, for the local file and for
`ggml-org/gpt-oss-20b-GGUF`. With 11, llama.cpp needs 7,030 MiB (6,500 weights + 114 KV + 416
compute, llama-completion at 4k) of the 7,106 MiB it sees free, 60 MiB left; verify matches
(KV +0.00, GPU weights -0.03, CPU +0.00). With 10 it needs 7,434 MiB and 0 MiB is left, so 11 is
the first that fits. llama-bench, tok/s:

| `--n-cpu-moe` | prompt | generate |
|---|---|---|
| 10 | 474.1 ± 28.7 | 17.9 ± 0.3 |
| 11 | 449.1 ± 13.0 | 16.4 ± 0.2 |
| 12 | 444.8 ± 2.1 | 14.9 ± 0.6 |

No drop at 11 that would show a spill. Every run today was about 15% slower than the first session
(10 gave 21.2 / 545 then), on AC power; not found why. The shown `<= 70` is far above 16.4.

**Qwen2.5-7B Q4_K_M at 131k: one layer off.** llama.cpp puts 14/29 layers on the GPU (2,172 MiB
weights, 3,328 MiB of the KV cache, 426 compute; 1,164 MiB left free). QuantPilot's share is
0.5155 of 29 = 14.95 layers. The KV cache matches (7,168.00). The speed bound went from `<= 45` to
`<= 29`; measured at 14 layers: 13.72 ± 0.46.

**vLLM (Qwen/Qwen2.5-1.5B-Instruct-AWQ, WSL2): fixed, one new gap.**
- At the default 0.92 vLLM fails as before (6.88 GiB free after its CUDA context), and QuantPilot
  now says so: Too Tight, "7.5 GB is free, about 6.9 GB once vLLM has started CUDA".
- The AWQ row is rated for vLLM, and its weights are 1.07 GB without the tied `lm_head` copy;
  vLLM logs `Model loading took 1.1 GiB`.
- At 0.8 vLLM starts every time: KV 4.5 GiB → 168,576 tokens at 4k; at 32k f16 4.47 GiB →
  167,408 tokens, fp8 4.47 GiB → 335,104 tokens (2.00×).
- **Gap:** `feasibility` has no way to rate at 0.8, so on an 8 GB card every vLLM row is Too Tight
  although vLLM serves it at 0.8 (also the AWQ / GPTQ rows of `meta-llama/Llama-3.2-1B-Instruct`).
  It could take `--gpu-memory-utilization`, or rate at the highest share that starts and name the
  flag, as it does for `--n-cpu-moe`.
- Small: the header then reads "1.47 of 0.00 GB (999%)".

**Gated repos: fixed.** `meta-llama/Llama-3.1-8B-Instruct` (not accepted) gives the new message
with the license link (`retest/llama-gated.txt`). Meta has accepted this account for Llama 3.2 1B
now; `feasibility` of it works.

**`ocq models`: fixed.** 10 models, all text generation (`retest/models.txt`).

## Mac side, after the re-test

- **vLLM on an 8 GB card:** when vLLM's default share is not free, or does not hold the model,
  `feasibility` now rates the vLLM rows at the highest lower share that starts, in 0.05 steps,
  and names it: "Rated at --gpu-memory-utilization 0.85, the highest that starts". On the
  laptop (7,689 MiB free) `Qwen/Qwen2.5-1.5B-Instruct-AWQ` is Perfect at 0.85, 1.47 of 6.80 GB.
  It also picks tensor parallel at a lower share when one busy card blocks the default.
- **Header:** "1.47 of 0.00 GB (999%)" now reads "needs 1.47 GB, none is free for it".
- **Re-test:** 0.80 was measured, 0.85 was not. Run vLLM at 0.85 (4k and 32k, f16 and fp8 KV)
  and check it starts and the KV token count, then `feasibility` of the AWQ and
  `meta-llama/Llama-3.2-1B-Instruct` AWQ / GPTQ rows.
- **Still open:** the Qwen 7B 131k layer split (14 placed, 14.95 predicted), item 6.

## NVIDIA PC re-test of `0ba7c29` (vLLM at the highest share that starts), 2026-11-08

`retest-2/`. 356 tests pass on Windows, ruff clean. `scripts/vllm_check.sh` takes `SHARE=0.85`.

**vLLM at 0.85: starts, fixed.** Qwen/Qwen2.5-1.5B-Instruct-AWQ, WSL2, 7,630-7,689 MiB free:

| run | vLLM | KV cache | tokens |
|---|---|---|---|
| default (0.92) | fails: 6.88 GiB free < 7.36 | - | - |
| 0.85, 4k | starts | 4.9 GiB | 183,552 |
| 0.85, 32k, f16 | starts | 4.87 GiB | 182,384 |
| 0.85, 32k, fp8 (Triton) | starts | 4.87 GiB | 365,056 (2.00×) |

4.87 GiB / 182,384 tokens = 28,671 B per token (vLLM prints GiB to two decimals), QuantPilot
28,672. vLLM sees 6.88 GiB free and 0.85 asks for 6.8 GiB: 0.08 GiB of room. QuantPilot's note
says "about 6.9-7.0 GB once vLLM has started CUDA".

`feasibility` of the AWQ repo at each setting: Perfect at 0.85, "1.47 of 6.80 GB (22%)" at 4k,
2.23 at 32k f16, 1.80 at 32k fp8, with "Rated at --gpu-memory-utilization 0.85, the highest
that starts".

**Llama-3.2-1B-Instruct AWQ / GPTQ rows: fixed.** Perfect (gpu) at 0.85, on Windows (`vLLM
v0.32.0` default, vLLM not installed) and in WSL2 (the installed vLLM's): AWQ 4-bit 1.33, GPTQ
4-bit 1.33, GPTQ 8-bit 1.81 GB. Before: Too Tight, unfit.

## Mac side: item 6 (layers on the GPU), branch `phase-4-layer-split`

QuantPilot now places whole layers as llama.cpp's `--fit` does (common/fit.cpp, b11429): the
output layer first, then blocks from the last one back, each with its weights and its share of
the cache, while 1 GiB stays free. A mixture of experts too big even without its experts keeps
them in RAM, except those of the lowest block on the GPU (OLMoE 64k: experts of blk.7-15 on
`CUDA_Host`, blk.6 on the GPU).

- `feasibility` gives the count for `gpu+ram` rows ("Q4_K_M: 14 of 29"), and the speed bound
  uses the share of the weights a token reads from those layers.
- `verify` compares the layer count and, now also when not every layer is on the GPU, the
  weights on each side. Replayed on all 39 saved logs: layers match in every run, Qwen 7B 131k
  (14/29, 2,172.17 / 2,288.28 MiB) and OLMoE 64k (11/17, 285.54 / 2,435.75 MiB) exactly.
  Left: Gemma 3's +0.04 MiB (item 7), the Granite q8_0 crash (item 8).
- `verify` predicts the layers with llama.cpp's own working memory from the log (the GPU
  compute buffer was 600 MiB for OLMoE at 64k partly offloaded, 178-183 MiB fully on the GPU);
  `feasibility` uses QuantPilot's llama-server estimate.
- **Re-test:** `ocq verify` on two new partly offloaded runs, e.g. Qwen2.5-7B Q4_K_M at 65,536
  and 98,304 tokens, and OLMoE at 32k with `-ctk q8_0`: check the "layers on the GPU" row.

## NVIDIA PC re-test of `0632e74` (layers on the GPU), 2026-11-09

`retest-3/`. 361 tests pass on Windows, ruff clean.

**Layers on the GPU: fixed, exact in every run.** `ocq verify`, layer count and weights on both
sides:

| run | layers | GPU weights MiB | CPU weights MiB |
|---|---|---|---|
| Qwen2.5-7B Q4_K_M 64k | 21 = 21 | 3,099.03 = 3,099.03 | 1,361.42 = 1,361.42 |
| Qwen2.5-7B Q4_K_M 96k | 17 = 17 | 2,564.50 = 2,564.50 | 1,895.95 = 1,895.95 |
| Qwen2.5-7B Q4_K_M 128k | 14 = 14 | 2,172.17 = 2,172.17 | 2,288.28 = 2,288.28 |
| OLMoE 64k (experts in RAM but blk.6's) | 11 = 11 | 285.54 = 285.54 | 2,435.75 = 2,435.75 |
| Jamba 3B 256k | 29 = 29 | 1,751.65 / 1,751.66 | 90.00 = 90.00 |

OLMoE 32k with `-ctk q8_0` fits whole (17/17), so it does not test the split; OLMoE 48k f16 and
64k q8_0 run with QuantPilot's `--n-cpu-moe` plan (12 and 1), and match too. `feasibility` gives
llama-server's own choice at all three contexts (llama-server `-v`, `server-qwen7b-*.log`):
"Q4_K_M: 21 / 17 / 14 of 29". QuantPilot's budget there is 6,083 MiB, llama.cpp's 6,082.

**New: a q8_0 cache on a model without grouped K/V heads needs much more working memory.**
The `--n-cpu-moe 1` plan for OLMoE at 64k, q8_0, overfills the card: 2,271 weights + 4,352 KV +
596 compute = 7,219 MiB with 7,106 free (0 left; Windows spills to RAM). QuantPilot's working
memory estimate is 208.8 MiB. llama-server reserves the same 596.09 MiB with every layer on the
GPU, so it is the cache type, not the experts. GPU compute buffer, llama-server `--fit off -ngl 99`:

| model (K/V heads / heads, K/V width) | context | f16 cache | q8_0 cache |
|---|---|---|---|
| OLMoE (16/16, 2048) | 16k / 32k / 64k | 96.01 / 112.01 / - | 164.09 / 308.09 / 596.09 |
| Qwen1.5-0.5B-Chat Q2_K (16/16, 1024) | 8k / 16k / 32k | 32.51 / 40.51 / 56.51 | 50.05 / 90.05 / 170.05 |
| Qwen2.5-7B (4/28, 512) | 16k / 32k | 155.01 / 171.01 | 155.09 / 171.09 |
| Gemma 2 2B (4/8, 1024) | 4k / 8k | 80.02 / 88.02 | 80.55 / 88.55 |
| Gemma 3 1B (1/4, 256) | 16k / 32k | 68.02 / 84.02 | 68.55 / 84.55 |

Both models with as many K/V heads as heads (OLMoE, and Qwen1.5 downloaded to check it) fit
q8_0 = mask (context x 512 x 2 B) + 2 x context x K/V width x 2 B (a 16-bit copy of one layer's
K and V) + a constant, exactly at all six points: 20.09 MiB for OLMoE, 10.05 for Qwen1.5. The
models with grouped K/V heads add 0.08-0.53 MiB. Not found in llama.cpp's source which kernel
needs the copy; worth reading there before adding it to `working_bytes`.

## Mac side: a quantized cache's working memory, branch `phase-4-layer-split`

Found in llama.cpp b11429: `ggml_cuda_flash_attn_ext_get_alloc_size` (`ggml/src/ggml-cuda/fattn.cu`).
For a 512-token batch on Turing or newer the MMA attention kernel runs, which reads 16-bit K and
V, so with a q8_0 / q4_0 cache the compute buffer also holds a 16-bit copy of the layer's K and V
(context × K/V width × 2 B each). Models with grouped K/V heads have a small copy that stays under
their feed-forward peak, hence their +0.08-0.53 MiB.

QuantPilot now counts, for a quantized cache on a separate GPU, the larger of the feed-forward
peak and the attention one (the copy + 6 rows of hidden size × 512 tokens × 4 B; llama.cpp took
5.02 rows for both OLMoE and Qwen1.5). Every run in `retest-3/server-*` is at or under it, OLMoE
and Qwen1.5 with q8_0 by 2-8 MiB. Header of Qwen1.5-0.5B-Chat Q2_K saved for the tests.

- OLMoE Q2_K at 64k, q8_0: `--n-cpu-moe 1` → **3**. From the header: 2 would need 7,080 MiB
  (26 left of 7,106), 3 needs 6,941 (165 left). QuantPilot takes 3 because it counts the CUDA
  context as 0.55 GB, 59 MiB more than this card's.
- f16 rows and Metal (Mac) are unchanged: Metal reads q8_0 K/V in its kernels. Not measured.
- **Re-test:** `ocq feasibility` of OLMoE at 65,536 with `--kv-cache q8_0` (expect
  `--n-cpu-moe 3`), then llama-server / `ocq verify` with it: no spill, about 165 MiB left.

## Mac side: the rest of the open items, branch `phase-4-layer-split`

- **Item 7 (Gemma +0.02-0.05 MiB): fixed.** llama.cpp's CUDA buffers pad every quantized
  tensor whose rows are not a multiple of 512 values by the rest of one such row, and align
  each tensor to 128 bytes (`ggml_backend_cuda_buffer_type_get_alloc_size`). Gemma 3 is 1,152
  wide, Gemma 2 2,304, gpt-oss 2,880. `verify` now matches every saved log in `verify/` within
  llama.cpp's 0.01 MiB rounding: Gemma 3 651.67 = 651.67, Gemma 2 1,323.26 = 1,323.26, gpt-oss
  8,522.54 = 8,522.54.
- **Item 8 (Granite MoE + q8_0 crash): worked around.** `ggml-impl.h:330` is the graph's hash
  table running full. A quantized cache turns on llama.cpp's K/V rotation (Hadamard, b11429),
  which adds nodes to every layer; a graph may have 8 × the model's tensors, and Granite 3.x MoE
  (many steps, few tensors) goes over. `LLAMA_ATTN_ROT_DISABLE=1` turns the rotation off and
  leaves the f16 graph, which ran. `verify` and the benchmark set it for `granitemoe` with a
  quantized cache; `feasibility` and `plan` name it.
- **Working memory when part of the model is in RAM: now predicted.** A layer whose cache is in
  RAM is brought to the GPU for attention: Qwen 7B 64k/96k/128k reserved 271/330/426 MiB, mask +
  that copy + 6 rows of activations exactly. The same rule as the q8_0 one; 8 rows are counted.
- **Mac (Metal) always reserves the copy** (`extra_kv_f16`, ggml-metal-ops.cpp) and a scratch
  buffer. Measured on the M1: Qwen1.5-0.5B 8k/16k/32k 58.21/98.22/178.23 MiB, QuantPilot 2 MiB
  over. Short sliding-window mask: sized to the window's cache now (Gemma 3 32k: 72 MiB over
  instead of 101).
- **MLA working memory: not the 754 MiB.** A 2-layer DeepSeek-V2-Lite with random weights,
  converted with b11429, reserves 243.40 MiB at 64k with llama-server on Metal (280.01 with
  llama-completion); QuantPilot says 307. Found instead: llama.cpp keeps the compressed MLA
  cache only when the GGUF has `key_length_mla`. Older conversions (one `attn_kv_b` tensor, as
  `mradermacher/DeepSeek-V2-Lite-Chat-GGUF`) run as plain attention with 9 × the cache;
  QuantPilot counted them compressed. Fixed. The 754 MiB was probably such a file.
- **Item 12: not reproduced.** The test is isolated (empty home folder, model cache, speed
  history), long temp paths do not break its table. If it fails again, save pytest's output.

**Re-test on the NVIDIA PC:**
1. `ocq verify` Granite 3.1 MoE Q2_K at 16k `--kv-cache q8_0`: it should run now (the log says
   "attention rotation force disabled"), compute buffer and weights compared.
2. `ocq verify` Gemma 3 1B at 4k: "weights on the GPU" ok, +0.00.
3. OLMoE Q2_K at 65,536 `--kv-cache q8_0`: `feasibility` `--n-cpu-moe 3`, and it runs with about
   165 MiB free.
4. `make check`.

## NVIDIA PC re-test of `d52efcc` and `d79b799`, 2026-11-09

`retest-4/`. ruff, ruff format and 390 tests pass on Windows. The full suite ran 40 times in a
row, also while the GPU runs below went on: 0 failures (item 12 not seen again).

1. **Granite 3.1 MoE Q2_K, 16k, q8_0: fixed.** Runs ("attention rotation force disabled
   (LLAMA_ATTN_ROT_DISABLE)"), KV 408.00, 25/25 layers, GPU weights 486.85, CPU 39.38, all
   +0.00. `feasibility` names the variable.
2. **Gemma 3 1B, 4k: fixed.** GPU weights 651.67 = 651.67.
3. **OLMoE Q2_K, 64k, q8_0:** `feasibility` said `--n-cpu-moe 2`, not 3 (see below). With 2,
   llama.cpp needs 7,080 of the 7,106 MiB it sees (10 left); llama-server with 3 runs with 759
   MiB free in nvidia-smi. Compute buffer 596.09 in both: the K/V copy is counted right now.
4. **All 9 GGUFs at 4k / 16k / 64k / 16k q8_0 (36 live runs, `verify/summary.txt`): every check
   ok**, 0 differ, Granite 3.1 MoE q8_0 included. Gemma 2 1,323.26 = 1,323.26, Gemma 3 651.67 =
   651.67 at every setting.

**Free memory on Windows: two different numbers, and QuantPilot uses neither.**
- CUDA reports 7,106 MiB free on this card in all 127 llama.cpp loads, and torch saw the same,
  while nvidia-smi showed 7.42-7.61 GiB free. With another process holding 2,048 MiB (nvidia-smi
  5,668 free) CUDA still said 7,106: under WDDM it is a fixed figure, not what is left.
  `llama-server --list-devices` prints it in 0.4 s ("CUDA0: ... (8187 MiB, 7106 MiB free)").
- llama.cpp's `--fit` and vLLM's start-up check work from that figure, so predicting them needs
  it; `verify` already reads it from the log. QuantPilot's nvidia-smi - 0.55 GB matched only
  while nvidia-smi showed about 7.5 GiB. Today it shows 7.61 GiB: QuantPilot's budget is 7,229
  MiB, 123 more, and the same gpt-oss 4k plan said `--n-cpu-moe` 10 during one run and 11 a few
  minutes later.
- **Going over CUDA's figure is not a spill here.** A CUDA context really takes about 90 MiB:
  torch's 2,048 MiB took 2,136 in nvidia-smi, llama-server's 7,127 MiB took 7,219 (574 MiB
  still free). llama-bench of gpt-oss (nvidia-smi 7,792 MiB free before):

  | `--n-cpu-moe` | GPU weights MiB | prompt tok/s | generate tok/s |
  |---|---|---|---|
  | 8 | 7,713 | 189.8 ± 4.6 | 18.59 ± 0.24 |
  | 9 | 7,309 | 491.5 ± 27.1 | 18.75 ± 0.22 |
  | 10 | 6,904 | 474.9 ± 7.7 | 17.82 ± 0.27 |
  | 11 | 6,500 | 457.3 ± 18.1 | 16.38 ± 0.52 |

  9 and 10 go over 7,106 and are the fastest; only 8, over nvidia-smi's free memory, falls (the
  prompt speed to 190). So for a `--n-cpu-moe` plan (no `--fit`) the limit is nvidia-smi's free
  memory less about 90 MiB, and QuantPilot's plans are conservative: 11 where 9 is fastest.
- **Correction to `retest-3`:** "0 left; Windows spills to RAM" for OLMoE 64k q8_0 with
  `--n-cpu-moe 1` was not shown. llama.cpp's 0 MiB is CUDA's figure; the 7,219 MiB it needed is
  under nvidia-smi's free memory less 90. Its speed was not measured.
- Suggested: CUDA's free figure (`--list-devices`, or cuMemGetInfo) for what `--fit` and vLLM
  will do; nvidia-smi's free memory less the measured context for what fits. On Linux without
  WDDM the two should agree; not measured here.

**verify's own runs of a `--n-cpu-moe` plan overfill CUDA's figure** (gpt-oss: 4k 7,434, 16k
7,318, 64k 7,256, 16k q8_0 7,129 MiB of 7,106): verify runs llama-completion, whose compute
buffer is 416 MiB, with a plan made for llama-server (203 / 219 / 267 MiB measured at 4k / 16k /
64k; QuantPilot estimates 211 / 223 / 271, at or over all three). llama-server with the same
plans: 4k 6,823 MiB, 16k 7,127, 64k 7,114.

## Mac side, after `retest-4`

- **Two free-memory figures (b44ab19).** QuantPilot reads CUDA's own free figure from the
  driver (`cuMemGetInfo` through ctypes, nvcuda.dll / libcuda.so.1, no torch) next to
  nvidia-smi's. llama.cpp's `--fit` and vLLM's start-up check are predicted from CUDA's figure;
  a `--n-cpu-moe` plan gets what is really free: on Windows nvidia-smi's free memory less 90
  MiB, on Linux CUDA's figure. gpt-oss at 4k with 7,792 MiB free in nvidia-smi now plans
  `--n-cpu-moe 9` (the fastest in your table), Qwen 7B 131k keeps 14 layers whatever
  nvidia-smi says. Without the driver reading, the old nvidia-smi - 0.55 GB stays.
- **`verify` runs llama-server** (4 conversations, `--no-warmup`), waits for "model loaded",
  stops it, and adds a row "working memory (upper bound)": QuantPilot's estimate against
  llama-server's GPU compute buffer + logits. On the M1, Qwen1.5-0.5B 8k/16k q8_0/32k: +1.6 to
  +1.9 MiB, about 2 s a run.

**Re-test on the NVIDIA PC:**
1. `uv run python -c "from ocq.hardware import cuda_memory_mib; print(cuda_memory_mib())"` on
   Windows and in WSL2: expect about `[(7106.0, 8187.5)]` (WSL2: about 7,045).
2. `ocq feasibility ggml-org/gpt-oss-20b-GGUF`: `--n-cpu-moe` from nvidia-smi's free memory
   (9 at about 7.8 GB free), then `ocq verify` of the local file with that plan: every row ok,
   and llama-bench speed as in your table.
3. `ocq verify` of the 9 GGUFs at 4k / 16k / 64k / 16k q8_0: the new working memory row ok in
   all (llama-server now), and Qwen 7B at 65,536 / 131,072: layers 21 / 14.
4. vLLM in WSL2 (AWQ repo): the note's "once vLLM has started CUDA" figure is CUDA's now.
5. `make check`.

## NVIDIA PC re-test of `b44ab19` and `4fee8c3`, 2026-11-09

`retest-5/`. ruff, ruff format and 395 tests pass on Windows (with the fix below).

1. **`cuda_memory_mib()`: works.** Windows `[(7106.0, 8187.5)]`; WSL2 `[(7096.0, 8187.5)]` from
   both venvs (not about 7,045: that is what vLLM sees after its own torch context).
2. **gpt-oss: fixed.** `feasibility ggml-org/gpt-oss-20b-GGUF` with 7,792 MiB free in nvidia-smi
   plans `--n-cpu-moe 9`; `verify` with it (llama-server) matches every row, working memory
   210.57 vs 206.06. llama-bench at 9: 484.6 ± 14.4 prompt, 18.30 ± 1.10 generate (491.5 / 18.75
   in the table before).
3. **9 GGUFs × 4 settings + Qwen 7B 64k / 128k (38 runs, `verify/summary.txt`): 35 ok, 3 failed.**
   Qwen 7B keeps 21 / 14 layers. Two failures were a verify bug, one is real:
   - **Bug, fixed here: llama-server's log lost lines.** `run_llama` read the log through the
     same `TemporaryFile` the server wrote to. An inherited handle shares the file position, so
     each `out.seek(seen)` could make the server's next write land over earlier lines. gpt-oss
     16k and OLMoE 16k q8_0 lost their `llama_context` / KV lines ("0 tokens, 1 conversation",
     KV 0 vs 408 MiB); Gemma 2 4k had a glued line but passed. Reproduced without llama.cpp
     (`retest-5/shared_handle.py`, one write per line): Windows 199,033 of 200,000 lines intact,
     Linux 198,547; with a separate read handle 200,000 on both. `run_llama` now gives the
     server its own write handle and reads through another (`verify.py`), and
     `test_the_server_log_is_read_whole_while_it_is_written` fails 3 of 3 times on the old code,
     passes on the new (Windows and WSL2). Rerun twice each (`verify-fixed/`): gpt-oss 16k (4
     ok, working memory 222.57 vs 222.06), OLMoE 16k q8_0 (5 ok, 176.77 vs 164.86), Gemma 2 4k
     (5 ok), no damaged lines.
   - **Real: Falcon-H1 16k q8_0, working memory 52.50 vs 53.17 (-0.67 MiB), both reruns.** Its
     f16 runs have 8.36 MiB of room (44.14 at 16k); q8_0 adds 9.03 MiB, about the K/V copy
     (16,384 × 128 × 2 B × 2 = 8 MiB). Falcon's estimate is the same for f16 and q8_0, so the copy
     seems to add to its peak here, not to replace it.
4. **vLLM 0.85 in WSL2: fixed.** Starts, 183,552 tokens; the note says "about 6.9 GB once vLLM
   has started CUDA" (7,096 MiB = 6.93 GiB), vLLM logs 6.88 GiB.
5. `make check`: see above.

**Other platforms, as far as this PC can test** (WSL2 Ubuntu; the GPU hidden by leaving
`/usr/lib/wsl/lib` off PATH; LFM2-350M Q4_0 at 4k, `retest-5/linux/`):

| | detection | `ocq models` | `feasibility` | `verify` |
|---|---|---|---|---|
| Linux + NVIDIA (`linux-cuda`) | RTX 4060, CUDA | ok | Perfect (gpu) | all 6 rows ok, with libgomp |
| Linux CPU only (`linux-cpu`, downloaded by QuantPilot) | no GPU | ok | Perfect (cpu) | **wrong**, see below |
| Windows CPU only | not testable: QuantPilot finds the GPU even without nvidia-smi on PATH | | | |

- **Both Linux builds need `libgomp.so.1`** (package `libgomp1`). This WSL has none (it is a
  minimal install; desktop Ubuntu usually has it, Docker and server images often not). The
  message is only "llama.cpp did not load the model; see its log"; the log says "error while
  loading shared libraries: libgomp.so.1". Suggested: catch that line and name the package.
- **CPU only, `verify` predicts a GPU.** llama.cpp's CPU build still logs "offloaded 17/17
  layers to GPU", and `verify` takes it: weights on the GPU 206.87 vs 0.00, CPU 52.50 vs 358.74.
- **CPU only, weights are repacked: `CPU_REPACK` 154.12 MiB** next to `CPU_Mapped` 204.62 MiB
  (the Zen 4 CPU backend repacks Q4_0 for its vector instructions). This 259 MiB file takes
  358.74 MiB of RAM, 38% more; `feasibility` does not count the copy.
- **Windows without nvidia-smi on PATH** still finds the card but shows "Free right now: 8.00
  GB VRAM", the total.
- Suggested: a setting to ignore the GPU (as `CUDA_VISIBLE_DEVICES=` does for CUDA programs), so
  the CPU-only path can be tested on any PC.
- Not testable here: native Linux (not WSL) with NVIDIA, AMD / Intel GPUs, Intel Macs, several
  GPUs.

## Mac side, after `retest-5`: phase 4 closed

- **Every `retest-5` log replayed with `84c853d`:** all ok, Falcon-H1 16k `q8_0` included (working
  memory 60.50 vs 53.17). The only misses are gpt-oss 16k and OLMoE 16k `q8_0` in `verify/`, the
  two logs damaged by the shared-handle bug; their `verify-fixed/` reruns pass. 405 tests, ruff
  clean on the Mac.
- **Mac, Metal, LFM2-350M Q4_0, 4k and 16k `q8_0`:** every row ok.
- **Mac, `OCQ_CPU_ONLY=1`: found and fixed one Arm-only case.** LFM2 has no `output.weight`, so
  llama.cpp loads `token_embd` a second time as the output layer (llama-model-loader.cpp,
  duplicated tensors). On Arm, Q6_K has a repack kernel, so that copy is repacked: `CPU_REPACK`
  206.62 MiB, 154.12 + the 52.50 MiB table. On x86 (no Q6_K kernel) it stays mapped, which is why
  the Ryzen saw 154.12. `verify.repacked(..., as_output=True)` counts it; now 206.62 vs 206.62 at
  4k and 16k `q8_0`. Test from the trimmed Mac log
  (`tests/tests/data/gguf/llama-server-lfm2-350m-4k-macos-cpu.log`).
- `retest-6/` (pushed alongside): its 28 runs are all ok, and replaying them with the Arm fix
  above changes nothing on x86 (the tied Gemma, Granite and LFM2 tables are Q6_K / Q8_0 there,
  with no x86 repack kernel). The README's CPU-only platform rows are backed by those logs.

Phase 4 is done on this branch.

## Fixed on the NVIDIA PC, 2026-11-09 (`retest-6/`)

Every item still open after `retest-5` is fixed and run here. `main` merged in (only the PR #18
merge commit). ruff, ruff format, 404 tests on Windows and on Linux (WSL2), and `pytest -m
slow` (real GGUF, AWQ, GPTQ, BNB, NVFP4 quantization on the RTX 4060): 5 passed.

| Item | Fix | Checked |
|---|---|---|
| Falcon-H1 16k q8_0, working memory 0.67 MiB under | A layer with both a cache and a Mamba state runs attention beside the Mamba block, so the K/V copy adds to the feed-forward peak (`llama_server_buffers_bytes(beside_state=)`). Only Falcon-H1 has such layers; Granite 4.0-H, Jamba and LFM2 have separate ones. | llama-server q8_0 over f16 at 4k / 16k / 64k: +4.03 / +9.03 / +33.03 MiB for a 2 / 8 / 32 MiB copy (`falcon-server/`). `verify`: 42.50 vs 36.17, 60.50 vs 53.17, 132.50 vs 125.17, ok |
| CPU only: `verify` predicted a GPU | A run with no GPU device and no GPU buffer is CPU only (the CPU build still logs "offloaded 17/17 layers to GPU"): every weight in RAM. | Linux CPU, 9 GGUFs × (4k f16, 16k q8_0): 18 of 18 ok; Windows CPU: 3 of 3 ok |
| CPU only: `CPU_REPACK` not predicted | `repacked()` follows `ggml_repack_get_optimal_repack_type` (b11429): type, the CPU features llama.cpp logs in `system_info`, rows a multiple of 8 (4 for some ARM kernels), never `token_embd`. Two rows: the mapped file range and the repacked copy. | All 9 GGUFs exact (LFM2 154.12, Qwen 7B Q4_K_M 1,364.24, OLMoE Q2_K 2,976.75 MiB with AVX-512); LFM2's 92 tensors are the ones llama.cpp names |
| "+38% RAM on CPU" (retest-5) | **Wrong, corrected.** The mapped buffer is address space over the file; the repacked tensors' pages are read once to make the copy and left to the OS. The RAM a model needs stays about its file size, so `feasibility` is unchanged. | |
| Benchmark after `optimize --execute` called a CPU run "gpu" | Same cause; it now looks for a model buffer on a GPU. A CPU run is recorded as `cpu` and kept out of the GPU speed history. | `test_a_benchmark_without_a_gpu_runs_on_the_cpu_and_is_recorded_so` |
| No way to test without a GPU | `OCQ_CPU_ONLY=1`: no GPU in the hardware profile (so the CPU build is downloaded and CPU rows planned), and llama.cpp gets `--device none` (`LLAMA_ARG_DEVICE=none`; llama-bench `-dev none`), so a Metal or CUDA build does not offload either. | Windows: `win-cpu` downloaded, `verify` ok; WSL2 with the GPU visible: CPU only, ok |
| Linux without libgomp: "did not load the model" | `missing_library()`: "llama.cpp cannot start: this system lacks libgomp.so.1. Install libgomp1 (Debian, Ubuntu), libgomp (Fedora) or gcc-libs (Arch)." Also macOS "Library not loaded" and Windows' DLL-not-found exit code; for `verify` and the quantize / benchmark steps. | WSL2 without the library: that message |
| Windows without nvidia-smi: free = total (8.00 GB) | The free memory comes from the CUDA driver when the card count and totals agree (6.94 GB); `wddm` stays off then, since nothing reads what is really left. | `test_without_nvidia_smi_the_free_memory_comes_from_cuda` |
| `results/retest-5/*.py` failed ruff | Rewritten; same result (Windows 198,996 of 200,000 lines with the shared handle, 200,000 with a separate one). | ruff clean |

GPU runs after the fixes (`windows-gpu/`): Falcon-H1 q8_0 4k / 16k / 64k, gpt-oss 4k, Qwen 7B
128k (14/29), OLMoE 64k q8_0, Gemma 2 16k q8_0: every row ok.

README: a **Platforms** section (what has been checked where, what is platform specific),
`OCQ_CPU_ONLY`, the CPU-only rows of `verify`, the libgomp packages.

Still not run anywhere: native Linux (outside WSL) with NVIDIA, AMD and Intel GPUs, Intel Macs,
several real GPUs.
