from __future__ import annotations

from contextlib import contextmanager

import pytest

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _CaptureCommunicator:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    @contextmanager
    def capture(self):
        self.events.append(f"enter:{self.name}")
        try:
            yield
        finally:
            self.events.append(f"exit:{self.name}")

    def close(self):
        self.events.append(f"close:{self.name}")


def _coordinator() -> GroupCoordinator:
    coordinator = object.__new__(GroupCoordinator)
    coordinator.ca_comm = None
    coordinator._graph_capture_communicators = {}
    return coordinator


def test_capture_only_communicators_enter_once_without_replacing_ca_comm():
    events = []
    coordinator = _coordinator()
    coordinator.ca_comm = _CaptureCommunicator("ca", events)
    first = _CaptureCommunicator("first", events)
    second = _CaptureCommunicator("second", events)

    assert coordinator.register_graph_capture_communicator("first", first) is first
    assert coordinator.register_graph_capture_communicator("first", first) is first
    assert coordinator.register_graph_capture_communicator("second", second) is second
    assert coordinator.get_graph_capture_communicator("first") is first

    with coordinator._capture_communicators():
        events.append("body")

    assert events == [
        "enter:ca",
        "enter:first",
        "enter:second",
        "body",
        "exit:second",
        "exit:first",
        "exit:ca",
    ]
    assert coordinator.ca_comm.name == "ca"


def test_capture_only_communicator_name_rejects_a_different_object():
    coordinator = _coordinator()
    first = _CaptureCommunicator("first", [])
    coordinator.register_graph_capture_communicator("fused", first)

    with pytest.raises(RuntimeError, match="already registered"):
        coordinator.register_graph_capture_communicator(
            "fused", _CaptureCommunicator("second", [])
        )


def test_destroy_closes_capture_only_communicator_before_process_groups(monkeypatch):
    events = []
    coordinator = _coordinator()
    coordinator.device_group = "device-group"
    coordinator.cpu_group = "cpu-group"
    coordinator.pynccl_comm = None
    coordinator.mq_broadcaster = None
    coordinator.register_graph_capture_communicator(
        "fused", _CaptureCommunicator("fused", events)
    )

    monkeypatch.setattr(
        "torch.distributed.destroy_process_group",
        lambda group: events.append(f"destroy:{group}"),
    )

    coordinator.destroy()

    assert events == [
        "close:fused",
        "destroy:device-group",
        "destroy:cpu-group",
    ]
    assert coordinator.get_graph_capture_communicator("fused") is None
