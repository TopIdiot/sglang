# WeLM Token-Owner 设计

## 1. 目标

本设计优化 WeLM 在 Attention 与 MoE 之间的数据布局，消除 Attention 外部重复的
hidden-state 通信，以及 Norm、Router 和 TopK 的重复计算。Token Owner 本质是 TP
lanes 之间沿 sequence/token row 维度执行的 sequence parallel：Attention 外部每个
row 只有一个 owner，Attention 内部仍保持完整 sequence 与原有 TP head/weight shard。

该能力独立于 DP Attention 和 Context Parallel。DP、CP、DP+CP 与纯 TP 使用同一套
owner layout、metadata 与 layer communicator，区别仅在 Global TP 中包含多少个
独立 Token Owner groups，以及每组送入 Attention 的 rows 属于完整 batch 还是
CP-local sequence shard。纯 TP 是 `owner_group_count = 1` 的特例，不是另一套实现。

目标拓扑包括：

```text
Global TP size = AttnDP size x AttnCP size x owner_group_size
owner_group_count = AttnDP size x AttnCP size
owner_group_size = AttnTP size

DP-only: owner_group_count = AttnDP size
CP-only: owner_group_count = AttnCP size
DP+CP:   owner_group_count = AttnDP size x AttnCP size
纯 TP:   owner_group_count = 1
```

以上分解以当前支持范围 `suffix_parallel_size = 1` 为前提。

Owner group 由运行时已有 TP group 推导，不能硬编码 TP2、TP4 或 TP8。DP
或 CP 开启时，该 group 是固定 `(dp_rank, cp_rank)` 的 Attention TP group；纯 TP 下
是 Global TP group。

代表性拓扑如下：

| 模式 | Global TP | AttnDP | AttnCP | Owner groups | TP lanes/group |
|---|---:|---:|---:|---:|---:|
| DP-only | 8 | 4 | 1 | 4 | 2 |
| CP-only | 8 | 1 | 4 | 4 | 2 |
| DP+CP | 16 | 2 | 2 | 4 | 4 |
| 纯 TP | 4/8/16/32 | 1 | 1 | 1 | 等于 Global TP |

本设计不通过降低精度、改变 Attention 数学语义或复制模型权重换取性能。

## 2. 术语与布局

### 2.1 通信域

- **Global TP group**：持有一份完整模型权重切分的所有 rank。
- **Token Owner group**：对一批完整 token rows 执行 TP-lane sequence parallel 的
  rank group；组内每个 TP lane 拥有连续且互斥的 rows。
- **Token domain**：一个 Token Owner group 共同处理的完整 token-row domain。
  DP-only 下是一个 DP rank 的完整 batch rows；CP 下是一个 `(dp_rank, cp_rank)` 的
  CP-local rows；纯 TP 下是全局唯一 batch 的完整 rows。
- **Owner lane**：Token Owner group 内的 TP rank，也是 owner layout 的一个 slot。
- **DP rank**：DP Attention 中处理一组请求和完整本地 token 序列的逻辑 rank。
- **Attention TP group**：固定 `(dp_rank, cp_rank)` 下切分 Attention heads/weights
  的 TP rank 集合；也是 DP/CP 拓扑中的 Token Owner group。

DP/CP 下 Token Owner group 等于固定 `(dp_rank, cp_rank)` 的 Attention TP group，
并要求 Global TP rank 顺序为 DP-major、CP-middle、Attention-TP-minor。纯 TP 下
Token Owner group 等于 Global TP group。所有拓扑都必须复用现有 process group，
Token Owner 不创建通信组。

### 2.2 TP-Lane Sequence Parallel 语义

Token Owner 只在 Attention 外部切分 sequence rows，不切 Attention heads、hidden
dimension 或模型权重。每层保持以下闭环：

1. 各 TP lanes 持有互斥的 owner-local rows，并在本地执行可逐 row 的 Norm、Router
   和 TopK；
2. 进入 Attention 前，在 Token Owner group 内 AllGatherV，令每个 TP weight/head
   shard 看到完整 token-domain rows；
3. O projection 后，在同一 group 内 ReduceScatterV，对 TP partial output 求和并按
   sequence rows 返回 owner；
4. 后续 hidden/residual 继续保持 owner-local，直到下一层 Attention。

因此纯 TP、DP 和 CP 下的组内算法完全相同。DP/CP 只让 Global TP 中并存多个
sequence-parallel domains，不参与组内 owner 算法。

Token Owner 与 CP 正交：Token Owner 在每个 CP rank 内的 TP lanes 之间切分
CP-local rows；进入 Attention 前只恢复当前 CP-local token domain。之后 CP 仍按原有
算法在 CP ranks 之间交换或聚合 Q/K/V/O。Token Owner 不恢复跨 CP 的完整 sequence，
也不改变 CP Attention 的数学过程和通信顺序。

### 2.3 三种逻辑布局

设计复用主线 `ScatterMode`：

- `TP_ATTN_FULL`：一个 token domain 的完整 token rows 在其所有 TP owner lanes 上
  复制；每个 lane 只持有自己的 Attention head/weight shard。
- `SCATTERED`：一个 token domain 的每个 token row 只属于一个 owner lane，
  即 token-owner 布局。
- `FULL`：Global TP group 的完整 token rows 在所有 Global TP ranks 上可见，
  用于未采用 Expert Parallel 的 Global-TP MLP/MoE 权重切分。

`SCATTERED` 只改变 hidden/residual 的物理 row 布局，不改变 request、token、
scheduler、KV cache 或 KV transfer 的所有权。

### 2.4 基础 owner 切分

对一个 token domain 的 `N` 个物理 model rows 和大小为 `A` 的 owner group：

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
必须恢复原始 token-domain rows。

## 3. 核心数据流

```text
上一层 owner-local hidden/residual
  -> Token-Owner-group AllGatherV(normalized hidden)
  -> token-domain complete rows x Attention head shard
  -> Attention + O projection partial output
  -> Token-Owner-group ReduceScatterV(sum partial O by token owner)
  -> owner-local O-Norm + residual + post-attention Norm
  -> owner-local Router + TopK
  -> Global-TP MoE 或 DeepEP MoE
  -> owner-local hidden/residual
  -> owner-local input Norm
  -> 下一层 Attention
```

第一层模型输入和最终模型输出仍遵守主线
`ScatterMode.model_input_output()` 的契约。最后一层只在 logits 路径确实需要时，
在 Token Owner group 内恢复完整 rows。

## 4. Attention 边界

### 4.1 进入 Attention

层间 hidden 和 residual 保持 owner-local。每个 lane 先在自己的 rows 上完成
input Norm，再通过 Token Owner group 的 `AllGatherV` 恢复完整 normalized hidden。

DP/CP 下，该通信只允许发生在当前 `(dp_rank, cp_rank)` 的 Token Owner group 内，
其他 DP 或 CP shards 不参与。纯 TP 下全局只有一个 Token Owner group，因此算法和
collective 次数不变，只是 group 覆盖全部 Global TP ranks。

CP 开启时，该 AllGatherV 只恢复 CP-local rows。随后每个 TP lane 独立进入现有 CP
Attention 路径，由 CP group 完成跨 CP shard 的 KV/Q/O 通信。两类 collective 的
process group 和职责不同，不能合并成 Global TP collective。

### 4.2 离开 Attention

O projection 在各 TP owner lanes 上产生同一批 token 的 partial O。目标
owner 必须收到该 token 在所有 head/weight shards 上的完整求和，因此执行
Token-Owner-group `ReduceScatterV`：

```text
owner_partial_o = ReduceScatterV(attention_partial_o, owner_sizes)
```

`ReduceScatterV` 完成后才能执行 O-Norm。O-Norm 是非线性操作，对每个 partial O
分别归一化再求和不等价。正确顺序必须是：

```text
sum partial O -> O-Norm -> residual add -> post-attention Norm
```

owner-local residual 必须与 ReduceScatterV 的 destination mapping 完全一致。
CP 开启时，必须先完成 CP Attention 对 CP-local Q rows 的输出合并，再进入 O
projection 和 Token Owner ReduceScatterV；不能把 TP partial sum 与 CP attention
的 LSE/output merge 混为一次归约。

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

多 group 拓扑下 Global TP gather/reduce-scatter 跨越 DP/CP 对应的 Token Owner
groups，服务于既有分片 expert 计算；纯 TP 下 Global TP 与唯一 Token Owner group
相同。纯 TP Phase 1 固定选择该路径，但只允许以下最小通信闭环：一次 owner-local
hidden/TopK `AllGatherV` 为所有 Global-TP expert weight shards准备完整输入，以及一次
expert partial output `ReduceScatterV`完成求和并直接返回原 owner。不得在两者之外
增加完整 hidden gather、MoE输出 all-reduce或 replicated layer output。

### 5.2 DeepEP

DeepEP 已经以 `SCATTERED` rows 为输入，并由 dispatch/combine 完成 expert
通信。Token-owner 路径必须复用主线 DeepEP 数据流：

- Router 和 TopK 在 source owner 上执行；
- dispatch 携带 owner-local token；
- combine 将结果返回 source owner；
- 不增加第二套 DeepEP communicator；
- 不改变 normal/low-latency/auto mode 的选择规则。

纯 TP、DP 与 CP 使用同一个 Router context 接口。差异只在 Global TP 中包含一个
还是多个 Token Owner groups，DeepEP 本身不感知拓扑类型。

DeepEP继续使用 owner-local dispatch/combine。纯 TP Phase 1可以支持该 backend，
但它不是 Global-TP MoE消除冗余通信的替代方案；`none` backend自身也必须满足
上一节的一次必要输入聚合加一次必要输出归约契约。

## 6. KV Mirror Contraction

KV mirror contraction 前使用基础连续 owner 切分。Contraction 后只保留每个请求
需要继续经过 mirror layers 的 survivor rows，这些 rows 必须留在 contraction
发生前的原始 lane，不能为了均衡重新分配。

因此 contraction 后的 owner sizes 可能高度不均衡，并允许空 lane。布局必须由
WeLM 已有 mirror metadata 推导，不能再次通过 token 数平均切分。

### 6.1 Forward Plan 与 Layer Identity

Token Owner 在每次 model forward 开始时只构造一次不可变的
`TokenOwnerForwardPlan`。该 plan 同时包含 contraction 前后的布局和 transition，
layer forward 期间不再根据已经执行到哪一层修改 Token Owner phase。

每个 decoder layer 在模型构造时根据 effective KV mirror pairs 和实际执行 layer
范围确定静态身份：

- `PRE_MIRROR`：第一个可执行 mirror target 之前的 layer；
- `MIRROR_BOUNDARY`：执行范围内第一个 effective mirror target，也是唯一允许执行
  Token Owner transition 的 layer；
- `MIRROR_TAIL`：boundary 之后继续执行的所有 layer，包括后续 mirror targets。

如果当前模型执行范围内没有 mirror boundary，所有执行 layer 都按 `PRE_MIRROR`
处理，plan 的 transition 为 `NONE`。多组 mirror pairs 也只允许第一个 boundary 改变
row 布局，后续 mirror targets 直接消费 `MIRROR_TAIL` 状态。

每个 layer 根据自己的静态身份，从当前 forward 的同一个 plan 中做 O(1) 查询。
查询只返回 plan 已预建的 layer state，不分配对象、不重新计算 layout，也不修改 plan。
Layer state 至少确定：

- Attention 输入和输出是否使用 Token Owner communicator；
- 输入使用 pre-contraction 还是 post-contraction layout；
- boundary 是否需要执行 survivor/residual contraction；
- Router、TopK 和 MoE 应使用的 physical/valid rows。

Transition 只有三种：

| Transition | `PRE_MIRROR` | `MIRROR_BOUNDARY` | `MIRROR_TAIL` |
|---|---|---|---|
| `NONE` | 保持当前 owner layout | 与前层相同 | 与前层相同 |
| `CONTRACT_KEEP_OWNER` | pre-layout | pre-layout 输入，post-layout 输出 | post-layout |
| `CONTRACT_EXIT_OWNER` | pre-layout | owner 输入，非 owner 输出 | 保持非 owner |

当前只有 ordinary Prefill 可以选择后两种 transition。Decode、MTP target verify、
NextN draft proposal 和 CUDA Graph replay 在本轮重构中均使用 `NONE`，保持各自已有
的 effective Token Owner 决策，不引入 mirror-tail transition。

`NONE` 仅表示本次 model forward 的 layer 序列中没有 Token Owner communicator
transition，不表示输入 rows 必须保持 scheduler 最初的数量。MTP/NextN 已有的 query
selection、row pruning 和 padding 必须在 plan 构造前完成；plan 将处理后的显式 counts
作为该 forward 的唯一初始 layout，并在所有执行 layer 中保持稳定。

Plan 只描述 Attention 外部 hidden/residual 的 row ownership 和相关 MoE metadata。
Q/K/V head layout、KV cache location、KV mirror 写入和 P/D KV transfer 不属于该 plan。

### 6.2 多 Group Global-TP MoE 的 mirror proxy

Global-TP MoE 需要全局 expert rows，但其他 Token Owner groups 的 survivor-lane
分布不可由本地平均 token 数推导。为避免新增 metadata collective，使用临时 proxy：

1. 每个 Token Owner group 按原 lane 顺序将 survivor rows 临时收集到
   lane 0；
2. Global TP MoE collective 只把该 proxy 作为本 group 的有效 source；
3. Expert 输出回到 proxy 后，在 Token Owner group 内还原到原始 lanes。

Proxy 只影响 MoE 通信 staging，不改变 survivor 的逻辑 owner，也不改变 KV cache。

### 6.3 DeepEP contraction

DeepEP 直接从 survivor 的原始 lane dispatch。有效 row mask 必须来自真实 owner
metadata，不能用 graph padding 后的物理 rows 或均匀切分值代替。

### 6.4 纯 TP contraction

纯 TP 只有一个 Token Owner group，且与 Global TP group 相同，因此不需要 DP
mirror proxy。Contraction 直接从原始 owner layout 推导 survivor owner sizes，
`local_layout` 与 `global_layout` 使用同一组 owner sizes。空 lane 继续以零 payload
参加 Token Owner group collective，不能把 survivor 重新平衡到其他 lane。

### 6.5 CP contraction

CP 下 survivor 不能跨 CP shards 迁移；它必须同时保留原始 CP rank 和原始 TP owner
lane。CP runtime 负责给出每个 CP shard 的有效 row domain，统一 Token Owner layout
只处理该 domain 内的 TP-lane owner sizes。CP shard 为空时，其 TP lanes 保持零 payload，
不能把 survivor 迁移到其他 CP rank 来做负载均衡。

## 7. 统一 Metadata 与 Forward Plan

`ForwardBatch` metadata 只提供 scheduler 已确定的事实；`TokenOwnerForwardPlan` 是
layer 执行语义的唯一来源。Layer 不能通过 tensor shape、tensor layout、动态属性是否
存在或已经修改过的 transport metadata 反推当前处于 mirror boundary 前还是后。

### 7.1 Metadata 来源

DP、CP 与纯 TP 复用同一套逻辑输入。owner-group-vector 中的 list slot 表示一个
Token Owner group，而不是固定表示 DP rank：

- pre-contraction physical rows：DP 使用 scheduler 同步后的
  `global_num_tokens_cpu`，纯 TP ordinary Prefill 使用显式 `extend_num_tokens`，CP 使用
  CP runtime 给出的 CP-local physical rows；
- immutable original rows：DP 使用 `original_global_num_tokens_cpu`；纯 TP/CP 直接将
  上述显式 pre-contraction counts 固化在 plan 中，不回写伪 DP metadata；
- contraction policy：DP 使用同步后的 `welm_kv_mirror_contract_flags`；单 group 纯 TP
  使用既有 forward mode、KV mirror、logprob 和 effective execution-range 条件；
- post-contraction logical rows：`welm_kv_mirror_output_size` 或对应 group 的 request
  counts；
- local survivor mapping：`welm_kv_mirror_last_q_indices_cpu` 和
  `welm_kv_mirror_active_batch_indices_cpu`；
- post-contraction physical/valid rows：由 logical rows 和已有 row-alignment/padding
  规则在 plan 构造时计算。

owner-group-vector 按 DP-major、CP-middle 顺序组织，长度为
`AttnDP size * AttnCP size`，当前 slot 由 `(dp_rank, cp_rank)` 唯一确定。DP-only
退化为每个 DP rank 一个 slot，CP-only 退化为每个 CP rank 一个 slot，纯 TP 长度为
一且当前 slot 为零。构造 plan 不新增 metadata collective，也不从 GPU tensor 执行
`.item()` 获取行数。

### 7.2 Plan 内容与生命周期

每个 model forward 的 `TokenOwnerForwardPlan` 在进入第一层前一次性构造，并绑定该
forward。Plan 由 `WeLMTokenOwnerRuntime` 持有，不作为新的 `ForwardBatch` 动态 phase
字段；runtime 必须校验 layer 查询对应当前绑定的 forward。纯 TP、DP 和 CP 的来源
metadata 在 plan 内归一化，不能为了复用 DP 分支而向纯 TP `ForwardBatch` 写入仅供
Token Owner 判断阶段的伪 DP metadata。Plan 至少预计算：

- effective Token Owner policy 和 `NONE`/`CONTRACT_KEEP_OWNER`/
  `CONTRACT_EXIT_OWNER` transition；
- pre/post local layout 和 global layout；退出 owner 时 post owner layout 可以为空，
  但 post physical/valid row counts 仍必须明确；
- 每个 owner group contraction 前后的 physical/valid rows；
- 本地 survivor mapping、valid-row mask 所需的 CPU indices；
- `PRE_MIRROR`、`MIRROR_BOUNDARY`、`MIRROR_TAIL` 三个预建 layer states。

Plan 构造完成后不可变。Mirror boundary 可以按 plan 的 post counts 原子发布
`global_num_tokens_cpu/gpu`、logprob/non-padded counts、`dp_padding_mode` 和
`global_dp_buffer_len` 等既有 DP transport view，使后续通信模块看到正确长度；该更新
不是 Token Owner phase 信号，也不能改变任何 layer state。后续 layer 仍只按自身
identity 查询 plan，不读取 marker 判断 contraction 是否已经发生。

混合 DP batch 中，所有 Global TP ranks 必须从同步后的 forward modes 和 contraction
flags 得到同一个 transition，以保持 collective 顺序一致。`CONTRACT_EXIT_OWNER` 的
communicator policy 作用于整个 forward，但只有 flag 为真的 owner groups 使用 post
contraction counts 和 survivor mapping；其他 groups 的 post counts 等于 pre counts。
不得根据当前 rank 的本地请求类型单独选择 transition。

因此实现中禁止：

- 用 `hidden_states.shape[0]`、`residual.shape[0]` 或二者是否相等选择 communicator；
- 用 `hasattr`、临时 contracted-row marker 或 mutable phase 表示执行进度；
- contraction 后失效并重算 layout cache；
- 在每层重新生成 owner sizes、survivor mapping 或 valid mask。

普通 Prefill 若缺少构造 transition 所需的任一 metadata，必须在进入第一层前 fail-fast，
不能执行到 mirror boundary 后再回退。Decode/MTP/NextN 只需构造稳定的 `NONE` plan，
但仍必须使用各自已有的显式 request/token/capture counts，不能把 tensor shape 作为语义
来源。

### 7.3 Scale-Seq

Scale-Seq 将每个逻辑 token 展开为多个物理 hidden rows。Token-owner 的计数单位
始终是进入 Transformer layer 的物理 rows，而不是 `input_ids` 的逻辑 token 数。
后续纯 TP 支持该能力时，scheduler/model-boundary 必须显式提供 embedding 展开后的
physical row count；DP Attention 继续使用其现有、已经过 Scale-Seq 调整的全局 row
counts；CP 使用 Scale-Seq 展开后的 CP-local physical row counts。任何路径都不能在
layer 内通过 hidden shape 补建 metadata。
最终 logits 收缩回逻辑 token 的行为保持主线语义，不属于 owner layout。

当前纯 TP Phase 1 不支持 Scale-Seq。ServerArgs 在读取 WeLM model config 后对
`scale_seq_times > 0` fail-fast；该限制不改变已支持 DP Token Owner 路径的行为。

### 7.4 Deferred Mirror P/D

Deferred mirror只改变 Prefill和Decode各自执行的 layer范围，不改变单层 Token
Owner语义，也不传输 hidden-state layout：

1. Deferred Prefill在 cutoff前照常执行 Token Owner层；source layer进入 Attention
   前已经恢复完整 token-domain rows，因此 relocated target K/V finalizer继续按原有
   TP head shard写入 KV pool；
2. cutoff前最后一个已执行 layer的 hidden/residual可以保持 `SCATTERED`，因为
   deferred Prefill返回 completion marker并丢弃最终 hidden，不需要额外 final gather；
3. Decode收到 seed token后开始独立 model forward，并从自己的 physical rows重新
   构造 Token Owner layout；不复用或传输 Prefill侧 owner metadata；
4. KV cache、deferred completion协议和 P/D KV transfer继续使用现有实现，不感知
   Token Owner。

Deferred Prefill 的 effective mirror targets 位于 `execution_end_layer` 之后，因此其
执行范围内不存在 `MIRROR_BOUNDARY`，forward plan 的 transition 为 `NONE`。Decode
seed 是新的 model forward，独立构造自己的 `NONE` plan，不继承 Prefill plan。

因此 deferred mirror 与纯 TP Token Owner 可以直接组合，不新增 layout 转换、通信或
scheduler 状态。Deferred 自身已有的 FA3、Mooncake、PP1、无 speculative/HiCache
等限制继续生效；纯 TP Token Owner 对 Scale-Seq 的 Phase 1 限制独立于 deferred。

## 8. CUDA Graph 与 MTP/NextN

CUDA Graph 使用固定物理 shape，而 token-owner 语义由有效逻辑 rows 决定。Graph
capture 执行 Python model forward 时，每个 capture bucket 从固定 token/request counts
构造自己的不可变 `NONE` plan；layer 查询结果和 physical owner layout 随计算图一同
固定。Graph replay 不重新执行 Python layer forward，因此不得要求 runtime 在每次 replay
重新构造或切换 plan：

- capture rows：graph 分配、owner split 和 collective ordering 使用的固定 rows；
- valid rows：通过 graph runner 已有的 `num_token_non_padded` 等固定地址 GPU buffer
  在 replay 前更新；
- owner sizes：由 capture rows 和运行时拓扑确定的固定 destination/source rows。

Plan 的结构、communicator policy 和 physical layout 仍不可变；动态 valid-row buffer
只是 graph input，不是 Token Owner phase。Graph runner 继续复用现有 bucket key 和
`can_run` 校验 topology、forward kind 和 physical rows，不新增 Token Owner 专属
signature。Replay 只更新既有输入/valid-row buffers，然后执行 `graph.replay()`，不能
通过修改 runtime 当前 plan 改变已捕获的 layer 行为。

Global-TP MoE 为保持 graph shape 固定，可以计算 padding rows，但这些结果必须在有效
输出边界被丢弃，不能影响 logits、logprob 或 sampling。DeepEP contraction 使用显式
valid mask，padding/non-survivor rows 不进入有效 expert routing。

普通 decode graph 的 bucket 选择按 topology 退化：多 owner-group 的 DP Attention
继续使用 `max(global_num_reqs_cpu)`；单 owner-group 的纯 TP 没有 DP scheduler metadata，
直接使用本地 `batch_size`。只有 DP Attention graph 受 `can_run_dp_cuda_graph` 控制，
纯 TP 不能因为缺少该 DP scheduler permission 而被拒绝。graph bucket 未捕获或 shape
不满足时仍必须 fail-fast，不能静默回到 eager。

Decode extend、MTP target verify 和 draft proposal 都将有效 token 按现有 flattened
row 顺序切分到 owners。它们可以继续执行已有的 KV mirror/MTP KV 逻辑，但不触发
ordinary Prefill 的 Token Owner mirror-tail transition。Idle batch 和空 lane仍需维持
CUDA Graph 与 collective 的固定调用顺序，但 payload 可以为空。

已有 DP Token Owner 拓扑保持已经支持的 CUDA Graph 与 MTP/NextN 行为。单 group
纯 TP 的 Prefill 固定走 eager，Decode 支持普通 CUDA Graph、MTP target verify 和
NextN draft proposal CUDA Graph；Global-TP (`none`) 与 DeepEP 均复用现有 graph
runner。单 group graph bucket 使用本地请求数，不依赖 DP scheduler metadata；纯 TP
layout counts 也不触发 DP MLP-sync preparation。DeepEP `auto` 在 eager Prefill 使用
normal mode，在 target/proposal graph 使用 low-latency mode；未量化 BF16 MoE 的
target/draft runner 使用 DeepGEMM。Piecewise Prefill CUDA Graph 保持关闭。CP 后续也
必须复用相同 graph metadata，不得按 topology 再定义 owner 状态。

## 9. 通信规则

- Attention 边界通信仅限当前 Token Owner group。DP Attention 下是 DP-local
  Attention TP group；CP 下是固定 `(dp_rank, cp_rank)` 的 Attention TP group；纯 TP
  下是唯一的 Global TP group。
- DP Attention 下除现有 scheduler/control 同步和分片 MoE 必需通信外，不新增
  cross-DP collective 或 global barrier。
- CP 下不新增 CP-wide owner collective；跨 CP 通信只由现有 CP Attention、MoE
  layout 转换和 scheduler/control 契约触发。
- 不参与某个计算域的 rank 不参与该通信域的 collective。
- variable collectives 必须支持 unequal sizes、zero-sized source/destination 和
  world-size one。
- size/layout metadata 优先由 CPU 侧已有 request/token metadata 推导，不能在每层
  热路径增加 GPU `.item()` 或强同步 collective。
- 选择 token-owner fast path 后，布局、topology 或 backend 不满足要求时必须
  fail-fast；不得静默回退到 replicated Router、Global TP hidden all-reduce 或其他
  dense 路径。

## 10. 数值与状态不变量

- Attention 输入始终是完整 token-domain rows 加当前 lane 的 Attention shard。
- CP 下 token domain 明确指 CP-local rows，不是请求的全局完整 sequence。
- O-Norm 输入始终是所有 Attention TP partial O 的完整和。
- hidden、residual、Router、TopK、expert input/output 始终对应同一 token owner。
- logical flattened-token order 在每个 Attention 边界保持不变。
- dtype、Router precision、TopK、sampling 和模型权重切分保持主线语义。
- 不引入 FP8 或其他有损精度路径。
- KV cache、KV mirror 写入、P/D KV transfer、radix cache 和 scheduler ownership
  不受 token-owner 布局影响。
- 未开启 `--enable-token-owner` 的 DP、纯 TP、CP 和其他路径不得发生行为、通信或
  性能变化。

## 11. 支持范围

当前已实现的 DP Token Owner 拓扑支持：

- WeLM DP Attention；
- `AttnCP = 1`；
- `AttnTP > 1`，从运行时 group size 泛化；
- `PP = 1`；
- DP-major、Attention-TP-minor rank ordering；
- MoE A2A backend 为 `none` 或 DeepEP；
- eager prefill/decode；
- KV mirror contraction；
- CUDA Graph 与 WeLM MTP/NextN。

单 group 纯 TP 拓扑按以下阶段扩展：

1. Phase 1：`TP > 1`、`DP = 1`、`AttnCP = 1`、`PP = 1`，MoE backend 为
   Global-TP (`none`) 或 DeepEP，支持 eager Prefill、eager/普通 CUDA Graph Decode、
   普通 PD 不分离部署和 KV mirror contraction，并兼容 deferred mirror
   P/D；Global-TP 路径必须满足无冗余通信契约；
2. Phase 2（已实现）：在同一 metadata/layout 契约上支持 MTP/NextN target verify
   和 draft proposal graph，不改变 Phase 1 的 owner 语义；piecewise Prefill CUDA
   Graph 仍不在支持范围内。

CP 集成是同一设计的后续 topology phase，而不是新功能模型：

- CP-only：`AttnDP = 1`、`AttnCP > 1`；
- DP+CP：每个 `(dp_rank, cp_rank)` 建立独立 Token Owner group；
- 复用 CP runtime 已有的 local row layout、rotation 和 CP Attention 通信；
- 将 CP 当前等价的 owner-local Router/TopK 与 TP-lane split 收敛到统一
  `TokenOwnerLayout` 契约，不保留第二套 owner 定义。

通过已有参数 `--enable-token-owner` 选择该能力，默认值为 `false`，不新增纯 TP
专用参数。开启后 runtime 从现有 TP groups 推导 group count、group index 和 lane
rank；不满足当前阶段支持条件时必须在模型安装或首次进入 fast path 前明确报错，
不能切回 replicated layout。WeLM model-specific ServerArgs 已默认关闭 piecewise CUDA
Graph，因此不新增参数或重复校验；纯 TP 仅在上述 backend 与 topology 边界内支持
现有 WeLM MTP/NextN，其他 speculative 组合继续 fail-fast。本文的 PD 混合部署均指
普通 PD 不分离。

参数名、metadata 和通用 layout/communicator 类型不包含 DP 或 CP 前缀。所有拓扑
继续复用同一参数和 layout contract，不重新定义用户接口。

以下场景不属于当前纯 TP 扩展：

- 非 WeLM V4 文本模型，包括当前尚未接入的 WeLM VLM；
- `AttnCP > 1`、`PP > 1`；
- 纯 TP 下的 Scale-Seq；
- suffix parallel、TBO、previous-precision；
- Router Replay、MK MoE Router；
- PDMux；
- DeepEP之外且主线未支持的 A2A backend；
- Prefill CP、DP scheduler 负载建模；
- KV cache、KV transfer、radix cache、HiCache 逻辑变更；
- 新 fused kernel 和 Router overlap；
- 纯 TP 中的 piecewise Prefill CUDA Graph；
- 当前纯 TP 实施阶段内的 CP runtime 收敛；CP 属于后续 topology phase。

不支持场景必须在 communicator 安装或首次进入 fast path 时明确报错。

## 12. 代码组织与规模

`welmv4.py` 只保留模型构造、layer forward 和 KV mirror 生命周期的必要接入，
相对主线新增目标约为 200 行。WeLM 专属的 capability 校验、owner layout 推导、
mirror survivor 映射和 Router context 放入单一 runtime 模块；通用 collective 继续
复用 `TokenOwnerLayout` 与 `LayerCommunicator`，不得复制通信实现。

Token-owner runtime 从现有通信组得到 owner-group count、group index、group size 和
lane rank。DP、CP 与纯 TP 必须共享 layout 生成、Router context、mirror survivor
映射和 forward plan；CP-specific runtime 只提供 CP-local rows/ordering，不重新实现
TP-lane owner 算法。不得新增第二套 `ForwardBatch` owner metadata、单独的 pure-TP
runtime 或复制现有多 group 逻辑。普通 decode graph 只允许在现有 graph runner 中
增加 singleton bucket 选择和 DP-only permission gate，不新增 Token Owner graph runner。
纯 TP Phase 1 不修改 scheduler、KV transfer 和 cache allocator。

runtime 在每次 model forward 开始时构造并安装一个不可变 plan；decoder layer 只保存
静态 layer identity，并通过 runtime 的 `state_for(layer_identity)` 返回 plan 中预建的
state。该查询不得分配对象或读取 mutable `ForwardBatch` phase。不得在 contraction 后
失效或重算 layout，不得使用动态 marker、tensor shape、多字段 identity cache 或第二份
valid-mask cache 判断执行阶段。重构必须删除旧的 state mutation、重复校验和包装逻辑，
不能在旧路径外再叠加一套 plan 分支，也不能仅通过机械搬文件满足行数目标。若
`welmv4.py` 新增超过 200 行，超出部分必须对应无法下沉的模型 forward 或 mirror
生命周期语义，并在提交前列出原因。

## 13. 验收标准

### 正确性

- 单元测试覆盖 odd/even token count、`N < AttnTP`、空 lane 和 idle rank；
- AttnTP2、AttnTP4、AttnTP8 使用同一实现；
- pure TP、DP-only、CP-only 与 DP+CP metadata 产生相同的 TP-lane
  sequence-parallel layout 语义；
- 后续启用纯 TP Scale-Seq 时使用 embedding 展开后的物理 row count；
- eager、CUDA Graph、MTP、cache hit 和 mirror contraction 遵循同一 owner contract，
  但只在对应 phase 的支持范围内启用；
- ordinary Prefill 的 pre/boundary/tail layers 在整个 forward 中读取同一个 plan；
  mixed DP batch 的所有 ranks 选择同一 transition，只有 flagged groups 改变 rows；
- Decode、MTP target verify、NextN draft proposal 和 deferred Prefill 使用 `NONE` plan，
  与重构前的 effective Token Owner、CUDA Graph 和 collective 行为一致；
- 缺失、长度不一致或越界的 row/survivor metadata 在第一层前明确报错，不能通过
  tensor shape 补建，也不能静默保留旧 communicator；
- 纯 TP `none`/DeepEP 的普通 decode、MTP target verify 和 NextN draft proposal graph
  均完成 capture/replay，padding bucket 不改变有效 token 输出，graph miss 明确报错；
- 对比确认 hidden、logits 以及前后若干 token logprob 与主线基线一致；
- AA-LCR 相对确认基线下降不超过 1 个百分点，且无异常 empty output。

### 通信与性能

- DP/CP trace 中 Token Owner 边界不存在跨 owner-group collective；
- Global-TP MoE 不恢复 replicated layer output；
- DeepEP 不重复 gather owner hidden；
- 纯 TP Global-TP MoE每层只保留必要的 expert-input AllGatherV和
  expert-output ReduceScatterV，不出现额外 all-reduce、重复 gather或完整输出复制；
- 分别报告 prefill throughput、TTFT、decode TPOT 和 output throughput；
- correctness 和 communication gate 通过后，性能结果才可用于接受优化。
