import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _make_model_runner(*, attn_cp_mode="none", enable_token_owner=False):
    server_args = SimpleNamespace(
        cuda_graph_bs=[1, 2, 4, 8, 12, 16],
        enable_two_batch_overlap=False,
        enable_torch_compile=False,
        torch_compile_max_bs=32,
        attn_cp_mode=attn_cp_mode,
        enable_token_owner=enable_token_owner,
    )
    return SimpleNamespace(
        server_args=server_args,
        req_to_token_pool=SimpleNamespace(size=2048),
    )


class TestCudaGraphBatchSizes(unittest.TestCase):
    def test_context_parallel_filters_odd_cuda_graph_bs(self):
        import sglang.srt.model_executor.cuda_graph_runner as cgr

        with (
            patch.object(cgr, "require_gathered_buffer", return_value=False),
            patch.object(cgr, "get_attention_tp_size", return_value=1),
            patch.object(cgr, "get_attention_cp_size", return_value=2),
        ):
            capture_bs, _ = cgr.get_batch_sizes_to_capture(_make_model_runner())

        self.assertEqual(capture_bs, [2, 4, 8, 12, 16])

    def test_sharded_kv_context_parallel_keeps_bs_one_cuda_graph(self):
        import sglang.srt.model_executor.cuda_graph_runner as cgr

        with (
            patch.object(cgr, "require_gathered_buffer", return_value=False),
            patch.object(cgr, "get_attention_tp_size", return_value=1),
            patch.object(cgr, "get_attention_cp_size", return_value=2),
        ):
            capture_bs, _ = cgr.get_batch_sizes_to_capture(
                _make_model_runner(attn_cp_mode="sharded-kv")
            )

        self.assertEqual(capture_bs, [1, 2, 4, 8, 12, 16])

    def test_token_owner_keeps_logical_request_buckets(self):
        import sglang.srt.model_executor.cuda_graph_runner as cgr

        with (
            patch.object(cgr, "require_gathered_buffer", return_value=True),
            patch.object(cgr, "get_attention_tp_size", return_value=4),
            patch.object(cgr, "get_attention_cp_size", return_value=1),
        ):
            capture_bs, _ = cgr.get_batch_sizes_to_capture(
                _make_model_runner(enable_token_owner=True)
            )

        self.assertEqual(capture_bs, [1, 2, 4, 8, 12, 16])

    def test_non_owner_keeps_attntp_bucket_alignment(self):
        import sglang.srt.model_executor.cuda_graph_runner as cgr

        with (
            patch.object(cgr, "require_gathered_buffer", return_value=True),
            patch.object(cgr, "get_attention_tp_size", return_value=4),
            patch.object(cgr, "get_attention_cp_size", return_value=1),
        ):
            capture_bs, _ = cgr.get_batch_sizes_to_capture(_make_model_runner())

        self.assertEqual(capture_bs, [4, 8, 12, 16])


class TestTokenOwnerCudaGraph(unittest.TestCase):
    def test_disabled_router_replay_has_no_buffers_or_dp_collective(self):
        import sglang.srt.layers.dp_attention as dp_attention
        from sglang.srt.layers.dp_attention import DpPaddingMode
        from sglang.srt.model_executor.cuda_graph_runner import DecodeInputBuffers

        buffers = DecodeInputBuffers.create(
            device=torch.device("cpu"),
            max_bs=2,
            max_num_token=2,
            hidden_size=4,
            vocab_size=8,
            dtype=torch.float32,
            dp_size=2,
            pp_size=1,
            is_encoder_decoder=False,
            require_mlp_tp_gather=True,
            seq_len_fill_value=1,
            encoder_len_fill_value=1,
            num_tokens_per_bs=1,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            welm_oe_decode_hash_num_branches=0,
            scale_seq_factor=1,
            enable_router_replay=False,
            router_replay_num_layers=2,
            router_replay_top_k=1,
        )
        self.assertIsNone(buffers.router_replay_topk_ids)
        self.assertIsNone(buffers.router_replay_mask)
        self.assertIsNone(buffers.router_replay_local_topk_ids)
        self.assertIsNone(buffers.router_replay_local_mask)
        forward_batch = SimpleNamespace(
            input_ids=torch.tensor([1]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
            out_cache_loc=torch.tensor([0]),
            positions=torch.tensor([0]),
            mrope_positions=None,
            encoder_lens=None,
            mamba_track_indices=None,
            mamba_track_mask=None,
            out_cache_loc_swa=None,
            router_replay_topk_ids=None,
            router_replay_mask=None,
            global_num_tokens_gpu=torch.tensor([1, 0], dtype=torch.int32),
            dp_padding_mode=DpPaddingMode.MAX_LEN,
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
        )

        with patch.object(
            dp_attention,
            "dp_gather_replicate",
            side_effect=AssertionError("inactive router replay communicated"),
        ):
            buffers.populate_from_forward_batch(
                forward_batch=forward_batch,
                raw_bs=1,
                raw_num_token=1,
                bs=1,
                seq_len_padding_value=1,
                require_gathered_buffer=True,
                num_tokens_per_bs=1,
                nsa_enable_prefill_cp=False,
                enable_num_token_non_padded_flag=False,
            )

    def test_padding_preserves_disabled_lora_metadata(self):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([1]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=1,
            positions=torch.tensor([0]),
            lora_ids=None,
        )
        model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_cuda_graph_seq_len_fill_value=lambda: 1
            )
        )

        batch._pad_inputs_to_size(model_runner, num_tokens=2, bs=2)

        self.assertIsNone(batch.lora_ids)
        self.assertEqual(batch.input_ids.shape[0], 2)

    def test_owner_tracks_non_padded_rows_without_ep(self):
        import sglang.srt.model_executor.forward_batch_info as fbi

        with patch.object(
            fbi, "get_moe_expert_parallel_world_size", return_value=1
        ):
            self.assertFalse(fbi.enable_num_token_non_padded())
            self.assertTrue(
                fbi.enable_num_token_non_padded(
                    SimpleNamespace(enable_token_owner=True)
                )
            )

    def test_owner_uses_global_request_bucket(self):
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        forward_batch = SimpleNamespace(global_num_reqs_cpu=[1, 8, 3, 0])

        self.assertEqual(
            CudaGraphRunner._requested_token_owner_graph_batch_size(forward_batch),
            8,
        )

    def test_owner_requires_global_request_counts(self):
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        with self.assertRaisesRegex(RuntimeError, "global request counts"):
            CudaGraphRunner._requested_token_owner_graph_batch_size(
                SimpleNamespace(global_num_reqs_cpu=None)
            )

    def test_owner_non_padded_rows_follow_uneven_attntp_split(self):
        import sglang.srt.model_executor.forward_batch_info as fbi

        actual = []
        for rank in range(4):
            with (
                patch.object(fbi, "get_attention_tp_size", return_value=4),
                patch.object(fbi, "get_attention_tp_rank", return_value=rank),
            ):
                local_rows = fbi.compute_local_num_token_non_padded(
                    torch.tensor(5, dtype=torch.int32),
                    6,
                )
            actual.append(int(local_rows))

        self.assertEqual(actual, [2, 2, 1, 0])


class TestAttnCPCudaGraphSeqBuckets(unittest.TestCase):
    def _make_runner(
        self,
        *,
        capture_bs=(1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 72),
        max_context_len=262144,
        max_total_num_tokens=1596282,
        max_prefill_tokens=16384,
        page_size=1,
        explicit=False,
        is_attn_cp_sharded_kv=True,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        runner = CudaGraphRunner.__new__(CudaGraphRunner)
        runner.dllm_config = None
        runner.capture_bs = list(capture_bs)
        runner.enable_seq_len_graph_buckets = True
        runner.disable_padding = False
        runner.attn_backend = SimpleNamespace(
            get_cuda_graph_seq_len_fill_value=lambda: max_context_len,
            max_context_len=max_context_len,
            page_size=page_size,
            cuda_graph_max_seq_len_is_explicit=explicit,
            is_attn_cp_sharded_kv=is_attn_cp_sharded_kv,
        )
        runner.model_runner = SimpleNamespace(
            max_total_num_tokens=max_total_num_tokens,
            server_args=SimpleNamespace(max_prefill_tokens=max_prefill_tokens),
        )
        return runner

    def _install_buckets(self, runner):
        buckets = runner._build_seq_len_fill_value_buckets_by_bs()
        runner.seq_len_fill_values_by_bs = buckets
        runner.seq_len_fill_value = max(max(values) for values in buckets.values())
        return buckets

    def test_sharded_kv_skewed_long_context_bucket_covers_target_concurrency(self):
        runner = self._make_runner()

        buckets = self._install_buckets(runner)

        self.assertIn(32, buckets)
        self.assertGreaterEqual(max(buckets[32]), 100000)
        self.assertIsNotNone(
            runner._select_seq_len_fill_value_for_bs(32, 100000),
            buckets[32],
        )

    def test_sharded_kv_full_context_fallback_uses_sparse_batch_frontier(self):
        runner = self._make_runner(
            capture_bs=(
                1,
                2,
                4,
                8,
                12,
                16,
                24,
                32,
                *range(40, 129, 8),
            )
        )

        buckets = self._install_buckets(runner)

        self.assertEqual(max(buckets[64]), 262144)
        self.assertLess(max(buckets[72]), 262144)
        self.assertEqual(max(buckets[128]), 262144)

    def test_sparse_skew_plan_bounds_graph_shape_count(self):
        # Match the production max-bs=512 capture list from the reproducer.
        # Skew buckets should grow with the logarithmic full-context frontier,
        # not with all 52 captured batch sizes.
        capture_bs = (
            1,
            2,
            4,
            8,
            12,
            16,
            24,
            32,
            *range(40, 257, 8),
            *range(272, 513, 16),
        )
        runner = self._make_runner(
            capture_bs=capture_bs,
            max_total_num_tokens=1765472,
            page_size=16,
        )

        full_context_bs = runner._build_full_context_fallback_batch_sizes()
        skew_bucket_bs = runner._build_skew_bucket_batch_sizes(full_context_bs)
        buckets = self._install_buckets(runner)

        self.assertEqual(
            full_context_bs,
            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512},
        )
        self.assertEqual(skew_bucket_bs, {12, 24, 40, 72, 136, 272})
        self.assertLessEqual(sum(len(values) for values in buckets.values()), 110)

    def test_full_context_fallback_uses_runtime_max_context_len(self):
        # Deliberately use a non-256K, non-page-aligned value to make sure the
        # fallback cap comes from the active model configuration, not a fixed
        # constant or an alignment-rounded approximation.
        runtime_max_context_len = 131071
        runner = self._make_runner(
            capture_bs=(1, 2, 4, 8, 16, 32, 64, 128),
            max_context_len=runtime_max_context_len,
            max_total_num_tokens=1765472,
            page_size=16,
        )

        buckets = self._install_buckets(runner)

        self.assertEqual(max(buckets[128]), runtime_max_context_len)
        self.assertEqual(
            runner._select_graph_shape(65, runtime_max_context_len),
            (128, runtime_max_context_len),
        )

    def test_sharded_kv_skew_bucket_covers_batch_above_long_bucket_cutoff(self):
        # Regression for new_eval: a real batch of 66 requests is padded to the
        # bs=72 graph.  The old hard cutoff only added skew buckets through
        # bs=64, so a 32K sequence could not select a graph and decode fell
        # back to eager even though aggregate KV usage was within capacity.
        runner = self._make_runner(
            capture_bs=(64, 72, 80, 88, 96, 104, 112, 120, 128),
            max_total_num_tokens=1765472,
            page_size=16,
        )

        buckets = self._install_buckets(runner)

        self.assertIsNotNone(
            runner._select_seq_len_fill_value_for_bs(72, 32768),
            buckets[72],
        )
        self.assertLess(max(buckets[72]), 262144)

    def test_sharded_kv_full_context_searches_larger_graph_batch(self):
        capture_bs = (
            1,
            2,
            4,
            8,
            12,
            16,
            24,
            32,
            *range(40, 513, 8),
        )
        runner = self._make_runner(
            capture_bs=capture_bs,
            max_total_num_tokens=1765472,
            page_size=16,
        )
        buckets = self._install_buckets(runner)

        self.assertEqual(runner._select_graph_shape(66, 262144), (128, 262144))
        self.assertEqual(runner._select_graph_shape(129, 262144), (256, 262144))
        self.assertEqual(runner._select_graph_shape(257, 262144), (512, 262144))
        for max_seq_len in (1, 16384, 32768, 131072, 262144):
            for raw_bs in range(1, 513):
                graph_bs, seq_cap = runner._select_graph_shape(
                    raw_bs, max_seq_len
                )
                self.assertIsNotNone(graph_bs, (raw_bs, max_seq_len))
                self.assertGreaterEqual(graph_bs, raw_bs, (raw_bs, max_seq_len))
                self.assertLessEqual(
                    graph_bs, 2 * raw_bs, (raw_bs, max_seq_len)
                )
                self.assertGreaterEqual(
                    seq_cap, max_seq_len, (raw_bs, max_seq_len)
                )

        # Keep the closest graph for the common skewed case instead of always
        # jumping to the full-context fallback frontier.
        self.assertEqual(runner._select_graph_shape(66, 32768), (72, 53504))
        self.assertLess(max(buckets[72]), 262144)

    def test_sharded_kv_capture_uses_one_shape_row_and_empty_dummy_rows(self):
        runner = self._make_runner(capture_bs=(128,))
        seq_lens = torch.empty(128, dtype=torch.int32)
        seq_lens_cpu = torch.empty(128, dtype=torch.int32)

        runner._prepare_seq_lens_for_capture(seq_lens, seq_lens_cpu, 262144)

        expected = torch.zeros(128, dtype=torch.int32)
        expected[0] = 262144
        self.assertTrue(torch.equal(seq_lens, expected))
        self.assertTrue(torch.equal(seq_lens_cpu, expected))

    def test_sharded_kv_medium_batch_uses_next_full_context_frontier(self):
        runner = self._make_runner(
            capture_bs=(
                1,
                2,
                4,
                8,
                12,
                16,
                24,
                32,
                *range(40, 129, 8),
            )
        )

        buckets = self._install_buckets(runner)

        self.assertLess(max(buckets[40]), 150000)
        self.assertEqual(runner._select_graph_shape(33, 150000), (64, 262144))

    def test_seq_len_bucket_alignment_rounds_up_to_preserve_coverage(self):
        runner = self._make_runner(
            capture_bs=(1,),
            max_context_len=4097,
            max_total_num_tokens=4097,
            max_prefill_tokens=1025,
            page_size=16,
        )

        buckets = self._install_buckets(runner)

        self.assertGreaterEqual(buckets[1][0], 1025)
        self.assertIsNotNone(runner._select_seq_len_fill_value_for_bs(1, 1025))

    def test_non_sharded_kv_does_not_expand_skew_buckets(self):
        runner = self._make_runner(
            is_attn_cp_sharded_kv=False,
        )

        buckets = self._install_buckets(runner)

        self.assertLess(max(buckets[40]), 150000)


if __name__ == "__main__":
    unittest.main()
