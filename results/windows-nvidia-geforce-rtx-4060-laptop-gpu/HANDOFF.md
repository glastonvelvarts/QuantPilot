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
