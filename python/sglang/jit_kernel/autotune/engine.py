"""Shared manifest and compile-queue mechanics from AttnCP verify autotuning."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Executor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Generic, Iterable, Iterator, Mapping, Protocol, TypeVar

_MANIFEST_SCHEMA_VERSION = 1


class EncodableKey(Protocol):
    def encode(self) -> str: ...


_Key = TypeVar("_Key", bound=EncodableKey)
_Job = TypeVar("_Job")
_Compiled = TypeVar("_Compiled")
_Winner = TypeVar("_Winner")


@dataclass(frozen=True)
class ManifestIdentity(Generic[_Key]):
    kernel_abi: str
    compiler_version: str
    cuda_version: str
    gpu_fingerprint: str
    topology_fingerprint: str
    dtype: str
    profile: str
    required_keys: tuple[_Key, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.kernel_abi,
                self.compiler_version,
                self.cuda_version,
                self.gpu_fingerprint,
                self.topology_fingerprint,
                self.dtype,
                self.profile,
            )
        ):
            raise ValueError("manifest identity fields cannot be empty")
        keys = tuple(self.required_keys)
        if not keys:
            raise ValueError("manifest must require at least one winner")
        encoded = tuple(key.encode() for key in keys)
        if any(not value for value in encoded) or len(encoded) != len(set(encoded)):
            raise ValueError("manifest workload keys must encode uniquely")
        normalized = tuple(
            key for _, key in sorted(zip(encoded, keys), key=lambda item: item[0])
        )
        object.__setattr__(self, "required_keys", normalized)

    def to_payload(self) -> dict[str, object]:
        return {
            "kernel_abi": self.kernel_abi,
            "compiler_version": self.compiler_version,
            "cuda_version": self.cuda_version,
            "gpu_fingerprint": self.gpu_fingerprint,
            "topology_fingerprint": self.topology_fingerprint,
            "dtype": self.dtype,
            "profile": self.profile,
            "required_keys": [key.encode() for key in self.required_keys],
        }


@dataclass(frozen=True)
class CompileCompletion(Generic[_Job, _Compiled]):
    index: int
    job: _Job
    compiled: _Compiled | None
    error: Exception | None


def production_compile_worker_count(
    cpu_count: int,
    participant_count: int,
) -> int:
    if participant_count <= 0:
        raise ValueError("participant_count must be positive")
    return max(1, min(16, cpu_count // participant_count))


def iter_bounded_compilation(
    jobs: Iterable[_Job],
    *,
    compile_fn: Callable[[_Job], _Compiled],
    max_workers: int,
    max_pending: int,
    executor_factory: Callable[..., Executor] = ThreadPoolExecutor,
) -> Iterator[CompileCompletion[_Job, _Compiled]]:
    if max_workers <= 0 or max_pending < max_workers:
        raise ValueError("max_pending must be at least max_workers")
    job_iter = iter(enumerate(jobs))
    pending = {}

    with executor_factory(max_workers=max_workers) as executor:

        def submit_available() -> None:
            while len(pending) < max_pending:
                try:
                    index, job = next(job_iter)
                except StopIteration:
                    return
                pending[executor.submit(compile_fn, job)] = (index, job)

        submit_available()
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                index, job = pending.pop(future)
                # Preserve the verify tuner's overlap: refill compilation before
                # domain-specific validation or benchmarking consumes the result.
                submit_available()
                try:
                    compiled = future.result()
                    error = None
                except Exception as exception:
                    compiled = None
                    error = exception
                yield CompileCompletion(
                    index=index,
                    job=job,
                    compiled=compiled,
                    error=error,
                )


def production_manifest_path(
    cache_root: str | Path,
    namespace: str,
    identity: ManifestIdentity[_Key],
) -> Path:
    if not namespace or namespace in (".", "..") or "/" in namespace:
        raise ValueError("manifest namespace must be one path component")
    digest = hashlib.sha256(
        json.dumps(
            identity.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return Path(cache_root) / namespace / f"{digest}.json"


def _read_manifest(
    path: Path,
    identity: ManifestIdentity[_Key],
    *,
    decode_winner: Callable[[object], _Winner],
) -> dict[_Key, _Winner] | None:
    try:
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION
            or payload.get("identity") != identity.to_payload()
        ):
            return None
        winner_payload = payload["winners"]
        if not isinstance(winner_payload, dict):
            return None
        expected = {key.encode(): key for key in identity.required_keys}
        if set(winner_payload) != set(expected):
            return None
        return {
            expected[encoded]: decode_winner(value)
            for encoded, value in winner_payload.items()
        }
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_manifest(
    path: Path,
    identity: ManifestIdentity[_Key],
    winners: Mapping[_Key, _Winner],
    *,
    encode_winner: Callable[[_Winner], object],
) -> None:
    payload = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "identity": identity.to_payload(),
        "winners": {
            key.encode(): encode_winner(winners[key]) for key in identity.required_keys
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_tune_manifest(
    path: str | Path,
    identity: ManifestIdentity[_Key],
    tune_fn: Callable[[], Mapping[_Key, _Winner]],
    *,
    encode_winner: Callable[[_Winner], object],
    decode_winner: Callable[[object], _Winner],
) -> dict[_Key, _Winner]:
    manifest_path = Path(path)
    cached = _read_manifest(
        manifest_path,
        identity,
        decode_winner=decode_winner,
    )
    if cached is not None:
        return cached
    winners = dict(tune_fn())
    if set(winners) != set(identity.required_keys):
        raise ValueError("autotune result is incomplete or malformed")
    _write_manifest(
        manifest_path,
        identity,
        winners,
        encode_winner=encode_winner,
    )
    return winners


def merge_autotune_winners(
    rank_winners: Iterable[Mapping[_Key, _Winner]],
    required_keys: Iterable[_Key],
    *,
    score_fn: Callable[[_Winner], tuple],
) -> dict[_Key, _Winner]:
    """Merge rank-local winners using a domain-provided deterministic score."""

    required = tuple(required_keys)
    encoded = tuple(key.encode() for key in required)
    if len(encoded) != len(set(encoded)):
        raise ValueError("required winner keys must encode uniquely")

    required_set = set(required)
    merged: dict[_Key, _Winner] = {}
    for winners in rank_winners:
        for key, winner in winners.items():
            if key not in required_set:
                continue
            current = merged.get(key)
            if current is None or score_fn(winner) < score_fn(current):
                merged[key] = winner

    missing = required_set - merged.keys()
    if missing:
        missing_by_encoding = sorted((key.encode(), key) for key in missing)
        raise RuntimeError(
            "autotune produced no winner for: "
            + ", ".join(encoded_key for encoded_key, _ in missing_by_encoding)
        )
    return {key: merged[key] for key in required}


def load_or_tune_coordinated_manifest(
    path: str | Path,
    identity: ManifestIdentity[_Key],
    *,
    local_tune_fn: Callable[[], Mapping[_Key, _Winner]],
    synchronize_cache_fn: Callable[
        [Mapping[_Key, _Winner] | None],
        Mapping[_Key, _Winner] | None,
    ],
    gather_fn: Callable[
        [Mapping[_Key, _Winner]],
        Iterable[Mapping[_Key, _Winner]],
    ],
    score_fn: Callable[[_Winner], tuple],
    encode_winner: Callable[[_Winner], object],
    decode_winner: Callable[[object], _Winner],
    publish: bool,
) -> dict[_Key, _Winner]:
    """Coordinate an exact manifest before tuning, then merge and publish winners."""

    manifest_path = Path(path)
    local_cached = _read_manifest(
        manifest_path,
        identity,
        decode_winner=decode_winner,
    )
    coordinated_cached = synchronize_cache_fn(local_cached)
    if coordinated_cached is None:
        local_winners = dict(local_tune_fn())
        winners = merge_autotune_winners(
            gather_fn(local_winners),
            identity.required_keys,
            score_fn=score_fn,
        )
    else:
        winners = merge_autotune_winners(
            (coordinated_cached,),
            identity.required_keys,
            score_fn=score_fn,
        )
    if publish and (local_cached is None or local_cached != winners):
        _write_manifest(
            manifest_path,
            identity,
            winners,
            encode_winner=encode_winner,
        )
    return winners
