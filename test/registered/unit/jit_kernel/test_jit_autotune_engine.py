import json
import threading
import time
from dataclasses import dataclass

import pytest

from sglang.jit_kernel.autotune.engine import (
    ManifestIdentity,
    iter_bounded_compilation,
    load_or_tune_coordinated_manifest,
    load_or_tune_manifest,
    merge_autotune_winners,
)


@dataclass(frozen=True, order=True)
class _Key:
    bucket: int

    def encode(self) -> str:
        return f"bucket:{self.bucket}"


@dataclass(frozen=True)
class _Winner:
    config: int
    latency_ms: float


def test_compile_queue_is_bounded() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def compile_fn(job: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return job * 2

    completions = tuple(
        iter_bounded_compilation(
            range(8),
            compile_fn=compile_fn,
            max_workers=2,
            max_pending=3,
        )
    )

    assert max_active == 2
    assert {completion.job for completion in completions} == set(range(8))
    assert all(completion.error is None for completion in completions)
    assert {completion.compiled for completion in completions} == {
        job * 2 for job in range(8)
    }


def test_compile_queue_refills_before_yielding_completed_job() -> None:
    second_compile_started = threading.Event()

    def compile_fn(job: int) -> int:
        if job == 1:
            second_compile_started.set()
        return job

    completions = iter_bounded_compilation(
        (0, 1),
        compile_fn=compile_fn,
        max_workers=1,
        max_pending=1,
    )
    first = next(completions)

    assert first.job == 0
    assert second_compile_started.wait(timeout=1)
    assert tuple(completions)[0].job == 1


def test_manifest_reuses_only_an_exact_complete_match(tmp_path) -> None:
    keys = (_Key(1024), _Key(4096))
    identity = _identity(keys)
    path = tmp_path / "winners.json"
    calls = 0

    def tune():
        nonlocal calls
        calls += 1
        return {
            key: _Winner(config=index, latency_ms=index + 0.25)
            for index, key in enumerate(keys)
        }

    first = load_or_tune_manifest(
        path,
        identity,
        tune,
        encode_winner=_encode_winner,
        decode_winner=_decode_winner,
    )
    second = load_or_tune_manifest(
        path,
        identity,
        tune,
        encode_winner=_encode_winner,
        decode_winner=_decode_winner,
    )
    assert first == second
    assert calls == 1

    payload = json.loads(path.read_text())
    payload["identity"]["kernel_abi"] = "stale"
    path.write_text(json.dumps(payload))
    assert (
        load_or_tune_manifest(
            path,
            identity,
            tune,
            encode_winner=_encode_winner,
            decode_winner=_decode_winner,
        )
        == first
    )
    assert calls == 2

    payload = json.loads(path.read_text())
    payload["winners"].pop(keys[-1].encode())
    path.write_text(json.dumps(payload))
    assert (
        load_or_tune_manifest(
            path,
            identity,
            tune,
            encode_winner=_encode_winner,
            decode_winner=_decode_winner,
        )
        == first
    )
    assert calls == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_manifest_is_replaced(tmp_path) -> None:
    key = _Key(512)
    identity = _identity((key,))
    path = tmp_path / "winners.json"
    path.write_text("{not-json")
    expected = {key: _Winner(config=1, latency_ms=1.0)}

    assert (
        load_or_tune_manifest(
            path,
            identity,
            lambda: expected,
            encode_winner=_encode_winner,
            decode_winner=_decode_winner,
        )
        == expected
    )
    assert json.loads(path.read_text())["winners"][key.encode()]


def test_merge_autotune_winners_uses_domain_score_and_requires_every_key() -> None:
    keys = (_Key(1024), _Key(4096))
    rank_winners = (
        {
            keys[0]: _Winner(config=3, latency_ms=1.0),
        },
        {
            keys[0]: _Winner(config=2, latency_ms=1.0),
            keys[1]: _Winner(config=5, latency_ms=2.0),
        },
    )

    merged = merge_autotune_winners(
        rank_winners,
        keys,
        score_fn=lambda winner: (winner.latency_ms, winner.config),
    )

    assert merged == {
        keys[0]: _Winner(config=2, latency_ms=1.0),
        keys[1]: _Winner(config=5, latency_ms=2.0),
    }
    with pytest.raises(RuntimeError, match="bucket:4096"):
        merge_autotune_winners(
            rank_winners[:1],
            keys,
            score_fn=lambda winner: (winner.latency_ms, winner.config),
        )


def test_coordinated_manifest_reuses_peer_cache_without_retuning(tmp_path) -> None:
    keys = (_Key(1024), _Key(4096))
    identity = _identity(keys)
    peer_cache = {
        key: _Winner(config=index, latency_ms=index + 0.25)
        for index, key in enumerate(keys)
    }
    calls = []

    winners = load_or_tune_coordinated_manifest(
        tmp_path / "peer-cache.json",
        identity,
        local_tune_fn=lambda: (_ for _ in ()).throw(
            AssertionError("peer cache must suppress local tuning")
        ),
        synchronize_cache_fn=lambda local: calls.append(
            ("synchronize", local is not None)
        )
        or peer_cache,
        gather_fn=lambda local: calls.append(("gather", local)) or (local,),
        score_fn=lambda winner: (winner.latency_ms, winner.config),
        encode_winner=_encode_winner,
        decode_winner=_decode_winner,
        publish=False,
    )

    assert winners == peer_cache
    assert calls == [("synchronize", False)]


def test_coordinated_manifest_cold_tunes_gathers_and_warm_loads(tmp_path) -> None:
    keys = (_Key(1024), _Key(4096))
    identity = _identity(keys)
    path = tmp_path / "coordinated.json"
    calls = []

    def local_tune():
        calls.append("tune")
        return {keys[0]: _Winner(config=1, latency_ms=1.0)}

    def gather(local):
        calls.append(("gather", tuple(local)))
        return (
            local,
            {keys[1]: _Winner(config=2, latency_ms=2.0)},
        )

    cold = load_or_tune_coordinated_manifest(
        path,
        identity,
        local_tune_fn=local_tune,
        synchronize_cache_fn=lambda local: calls.append(
            ("synchronize", local is not None)
        )
        or local,
        gather_fn=gather,
        score_fn=lambda winner: (winner.latency_ms, winner.config),
        encode_winner=_encode_winner,
        decode_winner=_decode_winner,
        publish=True,
    )
    warm = load_or_tune_coordinated_manifest(
        path,
        identity,
        local_tune_fn=lambda: (_ for _ in ()).throw(
            AssertionError("warm cache retuned")
        ),
        synchronize_cache_fn=lambda local: calls.append(
            ("synchronize", local is not None)
        )
        or local,
        gather_fn=lambda local: (_ for _ in ()).throw(
            AssertionError("warm cache gathered")
        ),
        score_fn=lambda winner: (winner.latency_ms, winner.config),
        encode_winner=_encode_winner,
        decode_winner=_decode_winner,
        publish=False,
    )

    assert cold == warm
    assert calls == [
        ("synchronize", False),
        "tune",
        ("gather", (keys[0],)),
        ("synchronize", True),
    ]


def _identity(keys: tuple[_Key, ...]) -> ManifestIdentity[_Key]:
    return ManifestIdentity(
        kernel_abi="kernel-v1",
        compiler_version="tvm-ffi-0.1.9",
        cuda_version="12.9",
        gpu_fingerprint="H20-sm90",
        topology_fingerprint="attntp2-nvlink",
        dtype="bfloat16",
        profile="balanced",
        required_keys=keys,
    )


def _encode_winner(winner: _Winner) -> dict[str, object]:
    return {
        "config": winner.config,
        "latency_ms": winner.latency_ms,
    }


def _decode_winner(payload: object) -> _Winner:
    assert isinstance(payload, dict)
    return _Winner(
        config=int(payload["config"]),
        latency_ms=float(payload["latency_ms"]),
    )
