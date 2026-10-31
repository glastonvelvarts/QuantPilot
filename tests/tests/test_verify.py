"""ocq verify: llama.cpp's own memory numbers against QuantPilot's prediction.

The log is llama-completion's (build b11429, -c 4096 -v) for granite-4.0-h-350m-Q2_K.gguf on an
Apple M-series Mac, trimmed to the memory lines. It has llama.cpp's trial pass (all buffers
0.00 MiB) and then the real load.
"""

from pathlib import Path

import pytest
from test_gguf_file import real_header

from ocq.gguf_file import config_from_gguf, hf_style_tensors
from ocq.model_analyzer import build_snapshot
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
    # buffer is the sum of its tensors, which for this file equals the whole data.
    header, _ = real_header(STEM)
    snap = build_snapshot(STEM, {}, config_from_gguf(header), hf_style_tensors(header), None)
    checks = predict(snap, header, parse_log(LOG), "f16", 0, unified_memory=False)
    gpu = next(c for c in checks if c.name == "weights on the GPU")
    assert gpu.predicted_mib == pytest.approx(sum(t.nbytes for t in header.tensors) / 2**20)
