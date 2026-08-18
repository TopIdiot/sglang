from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group, get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    PostProcessWeightsReqInput,
    PostProcessWeightsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    ReportShardTransferTargetReqInput,
    ReportShardTransferTargetReqOutput,
    InstallShardTransferPlanReqInput,
    InstallShardTransferPlanReqOutput,
    UpdateWeightsFromShardTransferReqInput,
    UpdateWeightsFromShardTransferReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:
    def flush_cache_after_weight_update(self: Scheduler, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        tp_success = success
        draft_runner = self._get_draft_model_runner()
        if success and draft_runner is not None:
            success, message = self.draft_worker.update_weights_from_disk(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
        if not success:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group.

        NOTE: only the target worker joins the NCCL process group. The draft
        worker does NOT join (it shares ``tp_rank`` with the target, so it
        cannot take a distinct rank in the same group, and a separate group
        would require trainer-side coordination). Instead the draft receives
        weights in-process through ``update_weights_from_distributed``, which
        mirrors how ``update_weights_from_tensor`` updates both models from one
        set of tensors. Therefore ``init``/``destroy`` do not touch the draft.
        """
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group.

        NOTE: see ``init_weights_update_group`` — the draft never joins the
        NCCL group, so there is nothing to destroy on the draft side.
        """
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter.

        The target receives the weights once via the NCCL process group, then
        both the target and the draft model are loaded in-process from the same
        received tensors. This mirrors ``update_weights_from_tensor`` and avoids
        the draft having to join the NCCL group (which is impossible: the draft
        shares ``tp_rank`` with the target, and the trainer allocates no extra
        ranks for it). The trainer is expected to broadcast the union of target
        and draft (e.g. MTP/NEXTN) parameters; each model's ``load_weights``
        skips parameter names it does not own.
        """
        success, message, received = self.tp_worker.receive_weights_from_distributed(
            recv_req
        )
        if not success:
            logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(success, message)

        # Load into the target model.
        success, message = self.tp_worker.load_weights_from_distributed(received)
        if not success:
            logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(success, message)

        # Load the same received tensors into the draft model in-process.
        draft_runner = self._get_draft_model_runner()
        if draft_runner is not None and received:
            success, message = self.draft_worker.load_weights_from_distributed(received)
            if not success:
                logger.error(message)
                return UpdateWeightsFromDistributedReqOutput(success, message)

        self.flush_cache_after_weight_update(recv_req)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        draft_runner = self._get_draft_model_runner()
        if recv_req.disable_draft_model or draft_runner is None:
            worker = self.tp_worker
        else:
            worker = self.draft_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        if success:
            self.flush_cache_after_weight_update(recv_req)
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        tp_success = success
        draft_runner = self._get_draft_model_runner()
        if success and draft_runner is not None:
            success, message = self.draft_worker.update_weights_from_ipc(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
        if not success:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def post_process_weights(self, recv_req: PostProcessWeightsReqInput):
        """Optional post-processing for updated weights (e.g., Marlin conversion)."""
        success, message = self.tp_worker.post_process_weights(recv_req)
        return PostProcessWeightsReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

            # Suspend each NCCL comm to release its GPU memory
            # (``ncclCommSuspend``, TCCL 2.30 / NCCL 2.30+). Sync first to
            # drain in-flight ops; dedup by id since tp / attn_tp / moe_*
            # may share one underlying PyNcclCommunicator.
            torch.get_device_module().synchronize()
            seen_pynccl_comms = set()
            for group in (
                get_tp_group(),
                get_attention_tp_group(),
                get_moe_ep_group(),
                get_moe_tp_group(),
            ):
                if group is None:
                    continue
                pynccl_comm = group.pynccl_comm
                if pynccl_comm is None:
                    continue
                if id(pynccl_comm) in seen_pynccl_comms:
                    continue
                seen_pynccl_comms.add(id(pynccl_comm))
                if not getattr(pynccl_comm, "available", False):
                    continue
                if not pynccl_comm.nccl.has_comm_suspend():
                    continue
                pynccl_comm.nccl_suspend()

        # ``ncclCommSuspend`` enqueues ``cuMemUnmap`` on NCCL's stream;
        # sync so downstream ``cudaMemGetInfo`` sees the release.
        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

            # Resume each suspended NCCL comm; same id-dedup as release.
            seen_pynccl_comms = set()
            for group in (
                get_tp_group(),
                get_attention_tp_group(),
                get_moe_ep_group(),
                get_moe_tp_group(),
            ):
                if group is None:
                    continue
                pynccl_comm = group.pynccl_comm
                if pynccl_comm is None:
                    continue
                if id(pynccl_comm) in seen_pynccl_comms:
                    continue
                seen_pynccl_comms.add(id(pynccl_comm))
                if not getattr(pynccl_comm, "available", False):
                    continue
                if not pynccl_comm.nccl.has_comm_suspend():
                    continue
                pynccl_comm.nccl_resume()

            # ``ncclCommResume`` enqueues ``cuMemMap`` on NCCL's stream;
            # sync so subsequent collectives see fully-mapped buffers.
            torch.get_device_module().synchronize()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            if recv_req.include_draft:
                payload = {
                    name: mr.check_weights(action=recv_req.action)
                    for name, mr in _iter_weight_model_runners(self)
                }
            else:
                payload = self.tp_worker.model_runner.check_weights(
                    action=recv_req.action
                )
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    # ---- Shard transfer control plane ----

    def report_shard_transfer_target(
        self: Scheduler, recv_req: ReportShardTransferTargetReqInput
    ):
        from sglang.srt.managers.shard_transfer_backend import (
            get_shard_transfer_backend,
        )

        try:
            backend = get_shard_transfer_backend()
            targets = dict(_iter_weight_model_runners(self))
            return ReportShardTransferTargetReqOutput(
                success=True,
                message="Success.",
                serialized_targets=backend.report_targets(targets),
            )
        except Exception as e:
            logger.warning(f"report_shard_transfer_target see error: {e}")
            traceback.print_exc()
            return ReportShardTransferTargetReqOutput(success=False, message=f"{e}")

    def install_shard_transfer_plan(
        self: Scheduler, recv_req: InstallShardTransferPlanReqInput
    ):
        from sglang.srt.managers.shard_transfer_backend import (
            get_shard_transfer_backend,
        )

        try:
            message = get_shard_transfer_backend().install_plan(
                recv_req.serialized_plan, self.tp_rank
            )
            local = {
                "success": True,
                "message": message,
            }
        except Exception as e:
            logger.warning(f"install_shard_transfer_plan see error: {e}")
            traceback.print_exc()
            local = {"success": False, "message": str(e)}

        results = _gather_shard_transfer_rank_results(self.tp_cpu_group, local)
        failures = _shard_transfer_failures(results, phase="install")
        if failures:
            return InstallShardTransferPlanReqOutput(
                success=False, message=" | ".join(failures)
            )
        return InstallShardTransferPlanReqOutput(
            success=True, message=results[0]["message"]
        )

    def update_weights_from_shard_transfer(
        self: Scheduler, recv_req: UpdateWeightsFromShardTransferReqInput
    ):
        from sglang.srt.managers.shard_transfer_backend import (
            get_shard_transfer_backend,
        )

        phase = "update"
        try:
            backend = get_shard_transfer_backend()
            per_target = {}
            for name, mr in _iter_weight_model_runners(self):
                per_target[name] = backend.update_weights(
                    name, mr, recv_req.serialized_source
                )
            if recv_req.flush_cache:
                phase = "flush"
                self.flush_cache_after_weight_update(recv_req)
            local = {
                "success": True,
                "message": "Success.",
                "per_target": per_target,
            }
        except Exception as e:
            logger.warning(f"update_weights_from_shard_transfer see error: {e}")
            traceback.print_exc()
            local = {
                "success": False,
                "message": f"{phase}: {e}",
                "per_target": {},
            }

        results = _gather_shard_transfer_rank_results(self.tp_cpu_group, local)
        failures = _shard_transfer_failures(results)
        if failures:
            return UpdateWeightsFromShardTransferReqOutput(
                success=False, message=" | ".join(failures)
            )

        entries, nbytes, seconds, per_target = _aggregate_shard_transfer_stats(results)
        return UpdateWeightsFromShardTransferReqOutput(
            success=True,
            message=f"pulled {entries} entries, {nbytes / 1e9:.2f}GB in {seconds:.2f}s",
            entries=entries,
            bytes=nbytes,
            seconds=seconds,
            per_target=per_target,
        )

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        draft_runner = self._get_draft_model_runner()
        if draft_runner is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            draft_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _iter_weight_model_runners(scheduler):
    """Yield the main and optional draft model runners.

    Draft resolution uses the scheduler's own ``_get_draft_model_runner`` so
    the receiver side never has to guess sglang's internal spec-decode layout.
    """
    yield "main", scheduler.tp_worker.model_runner
    draft_runner = scheduler._get_draft_model_runner()
    if draft_runner is not None and draft_runner is not scheduler.tp_worker.model_runner:
        yield "draft", draft_runner


def _gather_shard_transfer_rank_results(group, local: dict) -> list[dict]:
    results = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.all_gather_object(results, local, group=group)
    return results


def _shard_transfer_failures(
    results: list[dict], *, phase: str = "update"
) -> list[str]:
    return [
        f"{phase} rank {rank}: {result['message']}"
        for rank, result in enumerate(results)
        if not result["success"]
    ]


def _aggregate_shard_transfer_stats(
    results: list[dict],
) -> tuple[int, int, float, dict]:
    per_target = {}
    rank_seconds = []
    for result in results:
        rank_seconds.append(
            sum(stats["seconds"] for stats in result["per_target"].values())
        )
        for target, stats in result["per_target"].items():
            total = per_target.setdefault(
                target, {"entries": 0, "bytes": 0, "seconds": 0.0}
            )
            total["entries"] += stats["entries"]
            total["bytes"] += stats["bytes"]
            total["seconds"] = max(total["seconds"], stats["seconds"])

    entries = sum(stats["entries"] for stats in per_target.values())
    nbytes = sum(stats["bytes"] for stats in per_target.values())
    seconds = max(rank_seconds, default=0.0)
    return entries, nbytes, seconds, per_target


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
