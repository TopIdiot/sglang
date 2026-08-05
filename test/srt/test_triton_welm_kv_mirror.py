import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend


class TestTritonWeLMKVMirror(unittest.TestCase):
    def _backend(self):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=torch.tensor([0, 4, 8], dtype=torch.int32),
            max_extend_len=4,
        )
        backend.welm_kv_mirror_qo_indptr = torch.arange(3, dtype=torch.int32)
        return backend

    def test_uncontracted_query_keeps_draft_extend_metadata(self):
        backend = self._backend()
        forward_batch = SimpleNamespace(
            batch_size=2,
            welm_kv_mirror_contracted=False,
        )

        qo_indptr, max_extend_len = backend._get_extend_query_metadata(
            torch.empty(8, 12, 256), forward_batch
        )

        self.assertIs(qo_indptr, backend.forward_metadata.qo_indptr)
        self.assertEqual(max_extend_len, 4)

    def test_contracted_query_uses_one_query_per_request(self):
        backend = self._backend()
        forward_batch = SimpleNamespace(
            batch_size=2,
            welm_kv_mirror_contracted=True,
            custom_last_index=torch.tensor([3, 7]),
            kv_mirror_active_batch_indices=torch.tensor([0, 1]),
        )

        qo_indptr, max_extend_len = backend._get_extend_query_metadata(
            torch.empty(2, 12, 256), forward_batch
        )

        torch.testing.assert_close(
            qo_indptr, torch.tensor([0, 1, 2], dtype=torch.int32)
        )
        self.assertEqual(max_extend_len, 1)

    def test_contracted_query_rejects_attention_dp_subset(self):
        backend = self._backend()
        forward_batch = SimpleNamespace(
            batch_size=2,
            welm_kv_mirror_contracted=True,
            custom_last_index=torch.tensor([3]),
            kv_mirror_active_batch_indices=torch.tensor([1]),
        )

        with self.assertRaisesRegex(NotImplementedError, "attention-DP batch"):
            backend._get_extend_query_metadata(
                torch.empty(1, 12, 256), forward_batch
            )

    def test_cuda_graph_hook_validates_static_query_layout(self):
        backend = self._backend()
        backend.set_welm_mtp_mirror_cuda_graph_metadata(
            2, torch.tensor([0, 1, 2], dtype=torch.int32)
        )

        with self.assertRaisesRegex(RuntimeError, "Invalid WeLM KV mirror"):
            backend.set_welm_mtp_mirror_cuda_graph_metadata(
                2, torch.tensor([0, 1, 2], dtype=torch.int64)
            )


if __name__ == "__main__":
    unittest.main()
