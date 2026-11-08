"""ocq verify: llama.cpp's own memory numbers against QuantPilot's prediction.

The log is llama-completion's (build b11429, -c 4096 -v) for granite-4.0-h-350m-Q2_K.gguf on an
Apple M-series Mac, trimmed to the memory lines. It has llama.cpp's trial pass (all buffers
0.00 MiB) and then the real load.
"""

import sys
from pathlib import Path

import pytest
from test_gguf_file import real_header

from ocq import verify
from ocq.gguf_file import GgufTensor, config_from_gguf, hf_style_tensors
from ocq.model_analyzer import build_snapshot
from ocq.types import OcqError
from ocq.verify import parse_log, predict

LOG = (
    Path(__file__).parent / "data" / "gguf" / "llama-completion-granite-4.0-h-350m.log"
).read_text()
STEM = "granite-4.0-h-350m-Q2_K"


def test_only_the_real_load_is_read_not_the_trial_pass():
    log = parse_log(LOG)
    assert (log.context, log.conversations, log.layers_on_gpu) == (4096, 1, (33, 33))
    assert (log.gpu_mib, log.cpu_mib) == (149.07, 60.29)  # the trial pass says 0.00
    assert (log.kv_mib, log.state_mib) == (16.00, 21.57)


def test_pinned_ram_of_a_cuda_build_counts_as_ram():
    # llama.cpp's CUDA build, OLMoE on an RTX 4060: CUDA_Host is RAM the GPU copies through.
    cuda = (
        "load_tensors: loading model tensors, this can take a while... (load_mode = mmap)\n"
        "load_tensors: offloaded 17/17 layers to GPU\n"
        "load_tensors:    CUDA_Host model buffer size =     0.00 MiB\n"
        "load_tensors:   CPU_Mapped model buffer size =    50.00 MiB\n"
        "load_tensors:        CUDA0 model buffer size =  2410.10 MiB\n"
        "sched_reserve:      CUDA0 compute buffer size =   118.27 MiB\n"
        "sched_reserve:  CUDA_Host compute buffer size =    20.02 MiB\n"
    )
    log = parse_log(cuda)
    assert (log.gpu_mib, log.cpu_mib) == (2410.10, 50.00)
    assert (log.gpu_compute_mib, log.cpu_compute_mib) == (118.27, 20.02)


def test_a_llama_cpp_crash_is_reported_not_compared():
    crashed = (
        "load_tensors: loading model tensors, this can take a while...\n"
        "D:/a/llama.cpp/llama.cpp/ggml/src\\ggml-impl.h:330: fatal error\n"
    )
    with pytest.raises(OcqError, match="ggml-impl.h:330: fatal error"):
        parse_log(crashed)


def test_prediction_matches_the_log():
    header, _ = real_header(STEM)
    snap = build_snapshot(STEM, {}, config_from_gguf(header), hf_style_tensors(header), None)
    checks = predict(snap, header, parse_log(LOG), "f16", 0, unified_memory=True)
    assert [c.name for c in checks] == [
        "context cache (KV)",
        "Mamba / convolution state",
        "weights on the GPU",
        "weights on the CPU (file range)",
    ]
    assert all(c.ok for c in checks), [(c.name, c.predicted_mib, c.actual_mib) for c in checks]


def test_a_separate_gpu_holds_copies_not_a_file_range():
    # The embedding table (tensor 0) is tied here, so the GPU needs it too; with copies the GPU
    # buffer is the sum of its tensors as CUDA allocates them (cuda_bytes), here all of them.
    from ocq.verify import cuda_bytes

    header, _ = real_header(STEM)
    snap = build_snapshot(STEM, {}, config_from_gguf(header), hf_style_tensors(header), None)
    checks = predict(snap, header, parse_log(LOG), "f16", 0, unified_memory=False)
    gpu = next(c for c in checks if c.name == "weights on the GPU")
    assert gpu.predicted_mib == pytest.approx(sum(cuda_bytes(t) for t in header.tensors) / 2**20)
    assert gpu.predicted_mib - sum(t.nbytes for t in header.tensors) / 2**20 < 0.03


def test_working_memory_is_read_from_the_real_load():
    # Lines as llama.cpp b11429 printed them for Granite 4.0-H 350M at 4,096 tokens (Metal):
    # a trial pass that allocates nothing, then the real load.
    log = "\n".join(
        [
            "I load_tensors: loading model tensors, this can take a while... (load_mode = none)",
            "I sched_reserve:       MTL0 compute buffer size =   999.00 MiB",
            "I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
            "I llama_context:        CPU  output buffer size =     0.38 MiB",
            "I sched_reserve:       MTL0 compute buffer size =   206.01 MiB",
            "I sched_reserve:        CPU compute buffer size =    10.02 MiB",
        ]
    )
    parsed = parse_log(log)
    assert (parsed.gpu_compute_mib, parsed.cpu_compute_mib, parsed.output_mib) == (
        206.01,
        10.02,
        0.38,
    )


def test_a_partly_offloaded_model_is_compared_layer_by_layer():
    # Qwen2.5-7B Q4_K_M at 131k on the RTX 4060 Laptop (llama-completion b11429, --fit): 14/29
    # layers on the GPU. The file is the one QuantPilot converted there; bartowski's Q4_K_M has
    # the same tensors (its sizes give llama.cpp's 2,172.17 and 2,288.28 MiB).
    import gzip

    from ocq.model_analyzer import gguf_snapshot

    results = Path(__file__).parents[2] / "results" / "windows-nvidia-geforce-rtx-4060-laptop-gpu"
    log = gzip.decompress((results / "retest" / "qwen7b-131k-verify.log.gz").read_bytes())
    parsed = parse_log(log.decode("utf-8", errors="replace"))
    assert (parsed.layers_on_gpu, parsed.free_mib, parsed.gpus) == ((14, 29), 7106, 1)

    header, file_size = real_header("Qwen2.5-7B-Instruct-Q4_K_M")
    files = [("Q4_K_M", header.data_bytes)]
    snap = gguf_snapshot("Qwen2.5-7B-Instruct-Q4_K_M", {}, header, files, file_size, True)
    checks = predict(snap, header, parsed, "f16", 0, unified_memory=False)
    by_name = {c.name: c for c in checks}
    assert set(by_name) == {
        "context cache (KV)",
        "layers on the GPU",
        "weights on the GPU",
        "weights on the CPU (file range)",
    }
    assert all(c.ok for c in checks), [(c.name, c.predicted_mib, c.actual_mib) for c in checks]
    assert by_name["layers on the GPU"].predicted_mib == 14


@pytest.mark.parametrize(
    ("stem", "log_name", "layers"),
    [
        # A mixture of experts that does not fit even without its experts: --fit keeps them in
        # RAM but those of the lowest block on the GPU (blk.6), "offloaded 11/17 layers".
        ("olmoe-1b-7b-0924-instruct-q2_k", "olmoe-1b-7b-0924-instruct-q2_k-64k", (11, 17)),
        # Fits whole: the sliding-window layers' cache is sized for llama-completion's one
        # conversation, not llama-server's four, or the last layer seems not to fit.
        ("gemma-2-2b-it-IQ3_M", "gemma-2-2b-it-IQ3_M-64k", (27, 27)),
        # Gemma 3's rows are 1,152 wide: CUDA pads each quantized tensor to a multiple of 512
        # values and aligns it to 128 bytes, 651.67 MiB where the tensors add up to 651.63.
        ("gemma-3-1b-it-Q2_K", "gemma-3-1b-it-Q2_K-64k", (27, 27)),
    ],
)
def test_saved_nvidia_logs_match_layer_by_layer(stem, log_name, layers):
    import gzip

    from ocq.model_analyzer import gguf_snapshot

    results = Path(__file__).parents[2] / "results" / "windows-nvidia-geforce-rtx-4060-laptop-gpu"
    log = gzip.decompress((results / "verify" / f"{log_name}.log.gz").read_bytes())
    parsed = parse_log(log.decode("utf-8", errors="replace"))
    assert parsed.layers_on_gpu == layers
    header, file_size = real_header(stem)
    snap = gguf_snapshot(stem, {}, header, [("x", header.data_bytes)], file_size, True)
    checks = predict(snap, header, parsed, "f16", 0, unified_memory=False)
    assert all(c.ok for c in checks), [(c.name, c.predicted_mib, c.actual_mib) for c in checks]
    assert {c.name for c in checks} >= {"layers on the GPU", "weights on the GPU"}


def test_llama_servers_working_memory_is_checked_as_an_upper_bound():
    # ocq verify on the M1 (llama-server b11429, Metal, 4 conversations), Qwen1.5-0.5B-Chat Q2_K
    # at 32k, trimmed to the memory lines: QuantPilot 182.44 MiB of working memory, llama-server
    # 178.23 compute + 2.32 logits.
    stem = "qwen1_5-0_5b-chat-q2_k"
    text = Path(__file__).parent / "data" / "gguf" / "llama-server-qwen1_5-0_5b-32k-metal.log"
    log = parse_log(text.read_text())
    assert log.server and (log.context, log.conversations) == (32768, 4)
    header, _ = real_header(stem)
    snap = build_snapshot(stem, {}, config_from_gguf(header), hf_style_tensors(header), None)
    checks = {c.name: c for c in predict(snap, header, log, "f16", 0, unified_memory=True)}
    working = checks["working memory (upper bound)"]
    assert working.actual_mib == pytest.approx(178.23 + 2.32)
    assert working.ok and working.predicted_mib - working.actual_mib < 2
    assert all(c.ok for c in checks.values())
    # llama-completion's log (the Granite one above) is not llama-server's: no such row.
    assert not parse_log(LOG).server


FAKE_SERVER = r"""
import os, time
for i in range(20000):
    os.write(1, b"line %05d\n" % i)
os.write(1, b"srv  llama_server: model loaded\n")
time.sleep(60)
"""


def test_the_server_log_is_read_whole_while_it_is_written(tmp_path, monkeypatch):
    # A stand-in for llama-server, run as `python -m fake_server -c ...`: one write per line,
    # as llama.cpp's logger does. Reading the log through the handle the server writes with
    # made it write over lines (1,453 of 200,000 lost on Linux, 967 on Windows).
    (tmp_path / "fake_server.py").write_text(FAKE_SERVER)
    monkeypatch.setattr(verify.time, "sleep", lambda seconds: None)
    log = verify.run_llama(
        Path(sys.executable), Path("fake_server"), 4096, "f16", 0, {"PYTHONPATH": str(tmp_path)}
    )
    lines = log.splitlines()
    assert lines[:20000] == [f"line {i:05d}" for i in range(20000)]
    assert lines[20000] == "srv  llama_server: model loaded"


@pytest.mark.parametrize(
    ("platform", "repack_mib", "features"),
    [
        # Linux without a GPU (linux-cpu, Ryzen with AVX-512): Q4_0 repacked, the Q6_K embedding
        # table not (no x86 kernel for Q6_K).
        ("linux", 154.12, {"AVX2", "AVX512", "REPACK"}),
        # An M1 with OCQ_CPU_ONLY=1 (macos build, --device none): LFM2 has no output.weight, so
        # the Q6_K embedding table is loaded again as the output layer and repacked, +52.50.
        ("macos", 206.62, {"NEON", "DOTPROD", "REPACK"}),
    ],
)
def test_a_cpu_only_run_has_every_weight_in_ram_some_of_them_repacked(
    platform, repack_mib, features
):
    # ocq verify with llama-server b11429, trimmed to the memory lines. The CPU build still says
    # "offloaded 17/17 layers to GPU".
    stem = "LFM2-350M-Q4_0"
    data = Path(__file__).parent / "data" / "gguf"
    log = parse_log((data / f"llama-server-lfm2-350m-4k-{platform}-cpu.log").read_text())
    assert log.cpu_only and log.layers_on_gpu == (17, 17)
    assert (log.cpu_mib, log.repack_mib) == (204.62, repack_mib)
    assert features <= log.cpu_features
    header, _ = real_header(stem)
    snap = build_snapshot(stem, {}, config_from_gguf(header), hf_style_tensors(header), None)
    checks = {c.name: c for c in predict(snap, header, log, "f16", 0, unified_memory=False)}
    assert all(c.ok for c in checks.values()), [(c.name, c.predicted_mib) for c in checks.values()]
    assert "weights repacked for this CPU" in checks and "weights on the GPU" not in checks


def test_which_tensors_the_cpu_backend_repacks():
    x86, arm = frozenset({"REPACK", "AVX2"}), frozenset({"REPACK", "NEON", "DOTPROD"})

    def tensor(type_name, rows, name="blk.0.ffn_up.weight"):
        return GgufTensor(name, (1024, rows), type_name, 0)

    assert verify.repacked(tensor("Q4_0", 1024), x86)
    assert not verify.repacked(tensor("Q4_0", 1020), x86)  # rows must divide by 8 on x86
    assert verify.repacked(tensor("Q4_0", 1020), arm)  # by 4 with ARM's 4x4 kernels
    assert not verify.repacked(tensor("Q4_0", 1024), frozenset({"AVX2"}))  # built without it
    embd = tensor("Q6_K", 1024, "token_embd.weight")
    assert not verify.repacked(embd, arm)  # looked up, not multiplied
    assert verify.repacked(embd, arm, as_output=True)  # also the output layer when tied
    assert not verify.repacked(tensor("Q2_K", 1024), x86)  # needs AVX-512
    assert verify.repacked(tensor("Q2_K", 1024), x86 | {"AVX512"})
    assert not verify.repacked(tensor("Q6_K", 1024), x86) and verify.repacked(
        tensor("Q6_K", 1024), arm
    )


def test_a_missing_library_is_named_with_the_package_to_install():
    from ocq.quantizers.llama_cpp import missing_library

    # Linux without libgomp1 (WSL2, retest-5): both llama.cpp builds stop before any output.
    linux = (
        "/home/u/.cache/ocq/llama.cpp/b11429/linux-cpu/llama-server: error while loading shared "
        "libraries: libgomp.so.1: cannot open shared object file: No such file or directory"
    )
    assert "lacks libgomp.so.1. Install libgomp1 (Debian, Ubuntu)" in missing_library(linux)
    mac = "dyld[123]: Library not loaded: @rpath/libggml.dylib"
    assert "lacks @rpath/libggml.dylib." in missing_library(mac)
    assert "DLL" in missing_library("", 0xC0000135) and "DLL" in missing_library("", -1073741515)
    assert missing_library("load_tensors: offloaded 17/17 layers to GPU", 1) is None
    assert missing_library("", None) is None
