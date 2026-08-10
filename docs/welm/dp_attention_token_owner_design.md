# WeLM DP Attention Token-Owner 设计

## 1. 目标

本设计优化 WeLM 在 DP Attention 下 Attention 与 MoE 之间的数据布局，消除
Attention 外部重复的 hidden-state 通信，以及 Norm、Router 和 TopK 的重复计算。

Token-owner 是独立于 DP Attention 的通用数据布局能力。第一版仅实现和验证
WeLM DP Attention provider，但用户接口和通用通信抽象不得带 DP 限制，以便后续
扩展到 CP/TP 等其他合法拓扑。

目标拓扑为：

```text
Global TP = DP Attention x Attention TP
```

Attention TP 只描述 Attention 内部的 head 和权重切分，不等同于 Global TP。
Token-owner 布局必须由运行时 Attention TP group 推导，不能硬编码 AttnTP2、
AttnTP4 或 AttnTP8。

代表性拓扑如下：

| Global TP | Attention TP | DP Attention |
|---:|---:|---:|
| 8 | 2 | 4 |
| 8 | 4 | 2 |
| 16 | 2 | 8 |
| 16 | 4 | 4 |
| 16 | 8 | 2 |
| 32 | 2/4/8 | 16/8/4 |

本设计不通过降低精度、改变 Attention 数学语义或复制模型权重换取性能。

## 2. 术语与布局

### 2.1 通信域

- **Global TP group**：持有一份完整模型权重切分的所有 rank。
- **DP rank**：DP Attention 中处理一组请求和完整本地 token 序列的逻辑 rank。
- **Attention TP group**：同一个 DP rank 内切分 Attention heads 的 rank 集合。
- **Attention TP lane**：Attention TP group 内的一个 rank。

当前支持要求 Global TP rank 顺序为 DP-major、Attention-TP-minor。同一
Attention TP group 必须是已有 DP Attention 实现创建的 group，不能由
token-owner 路径重新创建。

### 2.2 三种逻辑布局

设计复用主线 `ScatterMode`：

- `TP_ATTN_FULL`：一个 DP rank 的完整 token rows 在其所有 Attention TP lanes
  上复制；每个 lane 只持有自己的 Attention head/weight shard。
- `SCATTERED`：一个 DP rank 的每个 token row 只属于一个 Attention TP lane，
  即 token-owner 布局。
- `FULL`：Global TP group 的完整 token rows 在所有 Global TP ranks 上可见，
  用于未采用 Expert Parallel 的 Global-TP MLP/MoE 权重切分。

`SCATTERED` 只改变 hidden/residual 的物理 row 布局，不改变 request、token、
scheduler、KV cache 或 KV transfer 的所有权。

### 2.3 基础 owner 切分

对一个 DP rank 的 `N` 个逻辑 token 和大小为 `A` 的 Attention TP group：

```text
base = N // A
remainder = N % A
owner_size[lane] = base + (lane < remainder)
owner_offset[lane] = sum(owner_size[0:lane])
```

每个 lane 持有连续 token rows。该定义必须正确覆盖：

- `N` 不能被 `A` 整除；
- `N < A`，部分 lane 为空；
- `N = 0` 的 idle rank；
- AttnTP2、AttnTP4、AttnTP8 以及其他合法 group size。

Owner 切分不改变 token 的逻辑顺序。对所有 owner rows 按 lane 顺序拼接后，
必须恢复原始 DP-local token rows。

## 3. 核心数据流

```text
上一层 owner-local hidden/residual
  -> Attention-TP-local AllGatherV(normalized hidden)
  -> DP-local complete tokens x Attention head shard
  -> Attention + O projection partial output
  -> Attention-TP-local ReduceScatterV(sum partial O by token owner)
  -> owner-local O-Norm + residual + post-attention Norm
  -> owner-local Router + TopK
  -> Global-TP MoE 或 DeepEP MoE
  -> owner-local hidden/residual
  -> owner-local input Norm
  -> 下一层 Attention
```

第一层模型输入和最终模型输出仍遵守主线
`ScatterMode.model_input_output()` 的契约。最后一层只在 logits 路径确实需要时，
在 DP-local Attention TP group 内恢复完整 rows。

## 4. Attention 边界

### 4.1 进入 Attention

层间 hidden 和 residual 保持 owner-local。每个 lane 先在自己的 rows 上完成
input Norm，再通过 DP-local Attention TP `AllGatherV` 恢复完整 normalized hidden。

该通信只允许发生在同一个 DP rank 的 Attention TP group 内。其他 DP ranks 不
参与，也不能引入 Global TP barrier。

### 4.2 离开 Attention

O projection 在各 Attention TP lanes 上产生同一批 token 的 partial O。目标
owner 必须收到该 token 在所有 head/weight shards 上的完整求和，因此执行
Attention-TP-local `ReduceScatterV`：

```text
owner_partial_o = ReduceScatterV(attention_partial_o, owner_sizes)
```

`ReduceScatterV` 完成后才能执行 O-Norm。O-Norm 是非线性操作，对每个 partial O
分别归一化再求和不等价。正确顺序必须是：

```text
sum partial O -> O-Norm -> residual add -> post-attention Norm
```

owner-local residual 必须与 ReduceScatterV 的 destination mapping 完全一致。

## 5. Router 与 Expert 布局解耦

主线 `LayerScatterModes` 当前只使用一个 `mlp_mode` 同时表示 Router 和 Expert
输入布局，但 Global-TP MoE 的正确优化需要将两者分开：

- `router_mode`：Router 和 TopK 的输入布局；
- `mlp_mode`：shared/routed expert 计算的输入布局。

默认模型保持 `router_mode == mlp_mode`，因此不改变现有行为。WeLM token-owner
路径的模式为：

| MoE backend | Router/TopK | Expert 输入 | residual/层间输出 |
|---|---|---|---|
| `none` | `SCATTERED` | `FULL` | `SCATTERED` |
| DeepEP | `SCATTERED` | `SCATTERED` | `SCATTERED` |

因此 `middle_residual_mode` 和非最终层 `layer_output_mode` 不能继续单纯由
`mlp_mode` 推导。Token-owner 激活时它们由 owner policy 决定；未激活时继续使用
主线现有推导规则。

### 5.1 Global-TP MoE (`none` backend)

Router 和 TopK 只计算 owner-local rows。完成路由后：

1. 将 owner-local hidden 和对应 TopK metadata 按 Global TP 顺序 `AllGatherV`；
2. 所有 Global TP ranks 对完整 expert rows 执行各自的 weight shard；
3. 将 routed expert 和 shared expert partial output 按原 owner mapping
   `ReduceScatterV`；
4. 输出直接回到原 token owner，不恢复成 replicated hidden。

Global TP 通信只服务于既有分片 expert 计算。Attention、Norm、Router、TopK 和
层间 residual 不得因此引入额外 cross-DP collective。

### 5.2 DeepEP

DeepEP 已经以 `SCATTERED` rows 为输入，并由 dispatch/combine 完成 expert
通信。Token-owner 路径必须复用主线 DeepEP 数据流：

- Router 和 TopK 在 source owner 上执行；
- dispatch 携带 owner-local token；
- combine 将结果返回 source owner；
- 不增加第二套 DeepEP communicator；
- 不改变 normal/low-latency/auto mode 的选择规则。

## 6. KV Mirror Contraction

KV mirror contraction 前使用基础连续 owner 切分。Contraction 后只保留每个请求
需要继续经过 mirror layers 的 survivor rows，这些 rows 必须留在 contraction
发生前的原始 lane，不能为了均衡重新分配。

因此 contraction 后的 owner sizes 可能高度不均衡，并允许空 lane。布局必须由
WeLM 已有 mirror metadata 推导，不能再次通过 token 数平均切分。

### 6.1 Global-TP MoE 的 mirror proxy

Global-TP MoE 需要全局 expert rows，但其他 DP group 的 survivor-lane 分布不可由
本地平均 token 数推导。为避免新增 cross-DP metadata collective，使用临时 proxy：

1. 每个 DP-local Attention TP group 按原 lane 顺序将 survivor rows 临时收集到
   lane 0；
2. Global TP MoE collective 只把该 proxy 作为本 DP group 的有效 source；
3. Expert 输出回到 proxy 后，在本地 Attention TP group 内还原到原始 lanes。

Proxy 只影响 MoE 通信 staging，不改变 survivor 的逻辑 owner，也不改变 KV cache。

### 6.2 DeepEP contraction

DeepEP 直接从 survivor 的原始 lane dispatch。有效 row mask 必须来自真实 owner
metadata，不能用 graph padding 后的物理 rows 或均匀切分值代替。

## 7. CUDA Graph 与 MTP/NextN

CUDA Graph 使用固定物理 shape，而 token-owner 语义由有效逻辑 rows 决定。实现
复用 capture bucket、全局 token/request counts 和 `num_token_non_padded`，不新增独立
的 physical-row layout 状态：

- capture rows：graph 分配、owner split 和 collective ordering 使用的固定 rows；
- valid rows：实际请求对应的 rows；
- owner sizes：由 capture rows 和运行时拓扑确定的固定 destination/source rows。

Global-TP MoE 为保持 graph shape 固定，可以计算 padding rows，但这些结果必须在有效
输出边界被丢弃，不能影响 logits、logprob 或 sampling。DeepEP contraction 使用显式
valid mask，padding/non-survivor rows 不进入有效 expert routing。

Decode extend、MTP target verify 和 draft proposal 都将有效 token 按现有 flattened
row 顺序切分到 owners。MTP/NextN 不建立另一套 owner 定义。Idle batch 和空 lane
仍需维持 CUDA Graph 与 collective 的固定调用顺序，但 payload 可以为空。

## 8. 通信规则

- Attention 边界通信仅限 DP-local Attention TP group。
- 除现有 scheduler/control 同步和分片 MoE 必需通信外，不新增 cross-DP
  collective 或 global barrier。
- 不参与某个计算域的 rank 不参与该通信域的 collective。
- variable collectives 必须支持 unequal sizes、zero-sized source/destination 和
  world-size one。
- size/layout metadata 优先由 CPU 侧已有 request/token metadata 推导，不能在每层
  热路径增加 GPU `.item()` 或强同步 collective。
- 选择 token-owner fast path 后，布局、topology 或 backend 不满足要求时必须
  fail-fast；不得静默回退到 replicated Router、Global TP hidden all-reduce 或其他
  dense 路径。

## 9. 数值与状态不变量

- Attention 输入始终是完整 DP-local token rows 加当前 lane 的 Attention shard。
- O-Norm 输入始终是所有 Attention TP partial O 的完整和。
- hidden、residual、Router、TopK、expert input/output 始终对应同一 token owner。
- logical flattened-token order 在每个 Attention 边界保持不变。
- dtype、Router precision、TopK、sampling 和模型权重切分保持主线语义。
- 不引入 FP8 或其他有损精度路径。
- KV cache、KV mirror 写入、P/D KV transfer、radix cache 和 scheduler ownership
  不受 token-owner 布局影响。
- 非 DP Attention 路径不得发生行为、通信或性能变化。

## 10. 支持范围

首个完整版本支持：

- WeLM DP Attention；
- `AttnCP = 1`；
- `AttnTP > 1`，从运行时 group size 泛化；
- `PP = 1`；
- DP-major、Attention-TP-minor rank ordering；
- MoE A2A backend 为 `none` 或 DeepEP；
- eager prefill/decode；
- KV mirror contraction；
- CUDA Graph 与 WeLM MTP/NextN。

通过独立参数 `--enable-token-owner` 选择该能力，默认值为 `false`。第一版
开启该参数时，WeLM 必须同时开启 DP Attention 并满足以上支持条件；否则在
安装阶段
明确报错。安装后遇到不支持的 forward mode 也必须报错，不能切回主线
replicated DP layout。

参数名和通用 layout/communicator 类型不包含 DP 前缀。第一版的 WeLM
provider、能力校验和测试矩阵可以明确限定 DP Attention。未来支持 CP/TP 时
复用同一参数和
layout contract，并增加对应 provider，不重新定义用户接口。

以下场景不属于首个版本：

- `AttnCP > 1`、`PP > 1`；
- suffix parallel、TBO、previous-precision；
- Router Replay、MK MoE Router；
- DeepEP 之外的 A2A backend；
- Prefill CP、DP scheduler 负载建模；
- KV cache、KV transfer、radix cache、HiCache 逻辑变更；
- 新 fused kernel 和 Router overlap；
- 非 DP Attention 的 CP/TP provider；它们复用同一接口，但不属于第一版能力集。

不支持场景必须在 communicator 安装或首次进入 fast path 时明确报错。

## 11. 验收标准

### 正确性

- 单元测试覆盖 odd/even token count、`N < AttnTP`、空 lane 和 idle rank；
- AttnTP2、AttnTP4、AttnTP8 使用同一实现；
- eager、CUDA Graph、MTP、cache hit 和 mirror contraction 遵循同一 owner contract；
- 对比确认 hidden、logits 以及前后若干 token logprob 与主线基线一致；
- AA-LCR 相对确认基线下降不超过 1 个百分点，且无异常 empty output。

### 通信与性能

- trace 中 Attention 边界不存在 cross-DP collective；
- Global-TP MoE 不恢复 replicated layer output；
- DeepEP 不重复 gather owner hidden；
- 分别报告 prefill throughput、TTFT、decode TPOT 和 output throughput；
- correctness 和 communication gate 通过后，性能结果才可用于接受优化。
