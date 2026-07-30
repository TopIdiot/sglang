import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
BENCHMARK_PATH = (
    REPO_ROOT / "benchmark" / "kernels" / "attntp_fused_norm" / "benchmark.py"
)


def _load_benchmark_module():
    module_name = "test_attntp_fused_norm_benchmark"
    module = types.ModuleType(module_name)
    module.__file__ = str(BENCHMARK_PATH)
    sys.modules[module_name] = module
    exec(
        compile(BENCHMARK_PATH.read_text(), str(BENCHMARK_PATH), "exec"),
        module.__dict__,
    )
    return module


def test_measure_batches_replays_and_flushes_l2_before_each_sample(monkeypatch):
    benchmark = _load_benchmark_module()
    log = []
    next_event_id = 0

    class FakeEvent:
        def __init__(self, *, enable_timing):
            nonlocal next_event_id
            assert enable_timing
            self.event_id = next_event_id
            next_event_id += 1

        def record(self):
            log.append(("event", self.event_id))

        def elapsed_time(self, _end):
            return float(self.event_id + 1)

    monkeypatch.setattr(benchmark.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "synchronize",
        lambda: log.append(("synchronize", None)),
    )
    monkeypatch.setattr(
        benchmark.dist,
        "barrier",
        lambda *, group: log.append(("barrier", group)),
    )

    def group_max_samples(samples, cpu_group):
        log.append(("group_max_samples", cpu_group))
        return samples

    monkeypatch.setattr(
        benchmark,
        "group_max_samples",
        group_max_samples,
        raising=False,
    )

    summary, samples = benchmark.measure(
        lambda: log.append(("operation", None)),
        lambda: log.append(("prepare", None)),
        lambda: log.append(("flush_l2", None)),
        warmup=2,
        iterations=3,
        cpu_group="cpu-group",
    )

    assert log == [
        ("barrier", "cpu-group"),
        ("prepare", None),
        ("flush_l2", None),
        ("operation", None),
        ("prepare", None),
        ("flush_l2", None),
        ("operation", None),
        ("synchronize", None),
        ("barrier", "cpu-group"),
        ("prepare", None),
        ("flush_l2", None),
        ("event", 0),
        ("operation", None),
        ("event", 3),
        ("prepare", None),
        ("flush_l2", None),
        ("event", 1),
        ("operation", None),
        ("event", 4),
        ("prepare", None),
        ("flush_l2", None),
        ("event", 2),
        ("operation", None),
        ("event", 5),
        ("synchronize", None),
        ("group_max_samples", "cpu-group"),
    ]
    assert samples == [1.0, 2.0, 3.0]
    assert summary.median_ms == 2.0


def test_replicated_baseline_uses_the_nccl_process_group(monkeypatch):
    benchmark = _load_benchmark_module()
    nccl_group = object()
    calls = []
    inputs = (
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.ones((2, 4), dtype=torch.float32),
        torch.ones((4,), dtype=torch.bfloat16),
        torch.ones((4,), dtype=torch.bfloat16),
    )

    monkeypatch.setattr(
        benchmark.dist,
        "all_reduce",
        lambda tensor, *, group: calls.append((tensor, group)),
    )
    monkeypatch.setattr(
        benchmark,
        "mmq_style_norm_after_attn",
        lambda *args: (args[0], args[1], torch.empty_like(args[1])),
    )

    prepare, operation, _result = benchmark.create_baseline_operation(
        inputs,
        nccl_group,
        mode=benchmark.OutputMode.REPLICATED,
        rank=0,
        world_size=4,
        owner=0,
    )
    prepare()
    operation()

    assert len(calls) == 1
    assert calls[0][1] is nccl_group


def test_microbenchmark_updates_only_production_runtime_metadata():
    benchmark = _load_benchmark_module()
    events = []

    class RecordedScalar:
        def __init__(self, name):
            self.name = name

        def fill_(self, value):
            events.append((self.name, value))

    actual_rows = RecordedScalar("actual_rows")
    lane_rotation = RecordedScalar("lane_rotation")

    benchmark.update_runtime_metadata(
        phase="decode",
        topology="tp",
        rows=17,
        actual_rows=None,
        lane_rotation=None,
        lane_rotation_value=0,
    )
    assert events == []

    benchmark.update_runtime_metadata(
        phase="prefill",
        topology="tp",
        rows=17,
        actual_rows=actual_rows,
        lane_rotation=lane_rotation,
        lane_rotation_value=0,
    )
    assert events == [("actual_rows", 17)]

    benchmark.update_runtime_metadata(
        phase="prefill",
        topology="cp",
        rows=17,
        actual_rows=actual_rows,
        lane_rotation=lane_rotation,
        lane_rotation_value=1,
    )
    assert events == [
        ("actual_rows", 17),
        ("actual_rows", 17),
        ("lane_rotation", 1),
    ]
