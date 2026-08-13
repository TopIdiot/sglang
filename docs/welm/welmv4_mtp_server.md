# WeLMV4 MTP Server 使用文档

本文说明如何用 SGLang server 启动带 WeLMV4 MTP 的模型。

## 适用模型

MTP serving 需要使用真正带 MTP/NextN 权重的 WeLMV4 checkpoint。模型配置需要满足：

- 主模型是 WeLMV4 MoE 模型。
- `config.json` 中 `num_nextn_predict_layers > 0`。

## MTP 参数

| 参数 | 含义 |
| --- | --- |
| `--speculative-algorithm NEXTN` | 开启 WeLMV4 MTP/NextN 推理。固定使用 `NEXTN` |
| `--speculative-draft-model-path` | MTP draft 权重路径。通常和 `--model` 使用同一个带 MTP 权重的 checkpoint。建议同 `--model` |
| `--speculative-num-steps` | draft 向前预测的深度。 |
| `--speculative-eagle-topk` | 每一步保留的候选分支数。`1` 是主推荐配置；`2` 可用于更宽的树，但吞吐不一定更高。 |
| `--speculative-num-draft-tokens` | 每轮 verify 的 draft token 数量上限。 |

## 相关 server 参数

| 参数 | 含义 |
| --- | --- |
| `--sampling-backend` | 选择采样后端，可选 `flashinfer`、`pytorch`、`ascend`。不设置时会自动选择：如果 FlashInfer 可用则使用 `flashinfer`，否则使用 `pytorch`。CUDA 上建议显式设置 `--sampling-backend flashinfer`；排障或 FlashInfer 不可用时再切到 `pytorch`。 |
| `--cuda-graph-max-bs` | CUDA graph 捕获的最大 batch size。未设置 `--cuda-graph-bs` 时，SGLang 会根据该值自动生成一组 capture batch sizes。值需要覆盖预期 decode 并发；越大显存占用越高。80A3 MTP c20 压测中使用 `20`。 |
| `--cuda-graph-bs` | 手动指定 CUDA graph 捕获的 batch size 列表，例如 `--cuda-graph-bs 1 2 4 8 16 20`。设置后会覆盖自动生成逻辑，并且 `cuda_graph_max_bs` 会取列表最大值。只在需要精确控制捕获列表、减少 graph 显存或对齐固定并发时使用。 |

## MTP 相关环境变量

| 环境变量 | 默认值 | 含义和限制 |
| --- | --- | --- |
| `SGLANG_ENABLE_SPEC_V2` | `1` | WeLMV4 MTP 必须使用 Spec V2/overlap schedule。建议启动命令里显式设置为 `1`；不要设为 `0`，也不要传 `--disable-overlap-schedule`。 |
| `SGLANG_WELM_MTP_SAMPLE_DRAFT` | 未设置 | `--speculative-eagle-topk 1` 时自动继承 verify 的采样语义：verify greedy 则 draft greedy；verify 随机采样则 draft 继承请求的 temperature、top-p、top-k。设为 `0` 强制 draft greedy；设为 `1` 显式启用该自动行为。tree proposal（`topk > 1`）不支持随机 draft sampling。 |
| `SGLANG_WELM_MTP_DRAFT_FIXED_TEMPERATURE` | 未设置 | 显式覆盖 draft temperature；未设置时继承 verify。值必须大于 `0`。 |
| `SGLANG_WELM_MTP_DRAFT_FIXED_TOP_P` | 未设置 | 显式覆盖 draft top-p；未设置时继承 verify。值必须在 `(0, 1]`。 |
| `SGLANG_WELM_MTP_DRAFT_SAMPLING_TOPK` | 未设置 | 正值会将 draft sampling 截断到前 K 个 token，从而显式覆盖 verify 的 top-k；未设置或 `<=0` 时继承 verify 的 top-k。值必须小于 vocab size。 |
| `SGLANG_WELM_V4D5_80A3_MTP_VERIFY_ATTENTION_BACKEND` | `fa3` | 只控制 WeLM V4D5 80A3 的 MTP target-verify attention。`fa3` 保持当前实现；`mk` 在满足下述固定契约时使用 MK verify kernel，不支持或启动自检失败时打印明确原因并回退 FA3。 |

### WeLM V4D5 80A3 MK verify attention

MK 路径是 target-verify 专用 fast path。Prefill、普通 decode 和 draft model
attention 始终使用原 FA3 backend，不属于 fallback。启用方式：

```bash
git submodule update --init 3rdparty/mk
git -C 3rdparty/mk submodule update --init ref/flashinfer
export SGLANG_WELM_V4D5_80A3_MTP_VERIFY_ATTENTION_BACKEND=mk
```

MK 必须包含已初始化的 `3rdparty/mk/ref/flashinfer` 子模块；缺失时
verify kernel 无法编译，启动日志会给出明确 fallback 原因。

MK 路径固定要求 H20/SM90、每个 DP replica 使用 TP4、BF16 KV cache、page size 16、
`steps=3`、`topk=1`、每请求 4 个 verify tokens，以及本地
Q/KV heads 为 6/1、head dim 256。只支持 Full attention 和 SWA512；
支持由独立 TP4 replica 组成的普通数据并行，但不支持 DP attention、
AttnCP、tree/custom mask 或 FP8 KV cache。

每个 worker 在 CUDA Graph capture 前自动运行 Full/SWA512 两个 MK-vs-FA3
数值检查。第一次真正调用 MK 时会打印 `Using MK WeLM V4D5 80A3...`；
任何安全的 fallback 都打印 `requested_backend=mk selected_backend=fa3`、
结构化 `reason`、`actual` 和 `expected`。同一原因只打印一次，但内部计数持续累加。

## 参数限制

必须满足这些限制，否则 server 会拒绝启动或结果没有经过验证：

- 必须启用 Spec V2：设置 `SGLANG_ENABLE_SPEC_V2=1`，并且不要传
  `--disable-overlap-schedule`。
- `--speculative-draft-model-path` 必须指向带 MTP 权重的 WeLMV4 checkpoint。
- 模型 ckpt config 中的 `num_nextn_predict_layers` 必须等于 `1` 或等于 `--speculative-num-steps`。
- 当 `--speculative-eagle-topk 1` 时，
  `--speculative-num-draft-tokens` 必须等于 `--speculative-num-steps + 1`。
  例如 `steps=3` 时必须设置 `draft_tokens=4`。
- 当 `--speculative-eagle-topk > 1` 时，
  `--speculative-num-draft-tokens` 不能超过
  `1 + topk + (steps - 1) * topk * topk`。例如 `steps=3, topk=2`
  时最大是 `11`。
- `topk > 1` 时不支持draft sampling，即不能设置 `SGLANG_WELM_MTP_SAMPLE_DRAFT=1`。
- 基础 Attention Backend 请固定使用 `fa3`。MK 环境变量只替换满足契约的
  target-verify attention，其余模式继续使用 FA3。

## 示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_ENABLE_SPEC_V2=1 \
python -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30002 \
  --model /home/jiayifeng/ckpts/80a3_mtp_3_sft_jfs \
  --served-model-name welmv4 \
  --trust-remote-code \
  --tp 4 \
  --mem-fraction-static 0.72 \
  --attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --enable-over-encoding \
  --enable-welm-kv-mirror-opt \
  --sampling-defaults openai \
  --sampling-backend flashinfer \
  --disable-radix-cache \
  --cuda-graph-max-bs 20 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /home/jiayifeng/ckpts/80a3_mtp_3_sft_jfs \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

## 请求示例

server 启动后可用 OpenAI compatible API 访问：

```bash
curl http://127.0.0.1:30002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "welmv4",
    "messages": [
      {"role": "user", "content": "写一段关于多 token prediction 的说明。"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```
