# WeLM Deferred Mirror Layer

## 功能概览

WeLM Deferred Mirror Layer 是面向 WeLM KV Mirror 结构的执行与调度优化。
它不改变模型的逻辑结果，而是把初始请求中最后一个 prompt token 的
mirror suffix 计算从 Prefill 延后到 Decode，使这部分计算进入更适合小行数、
动态合批和 CUDA Graph 的 Decode 执行路径。

开启该功能需要同时启用：

```bash
--enable-welm-kv-mirror-opt \
--enable-kv-mirror-deferred
```

该功能支持单体部署和已有的 P/D 分离部署。两种拓扑在 committed prefix
变为 `READY` 之后，共用同一套 seed Decode 生命周期。

## 相关概念

### Mirror layer

WeLM 的部分前层是 mirror source layer，部分后层是对应的 mirror target
layer。Source layer 的 QKV projection 除了生成本层 K/V，还会生成目标层需要的
mirrored K/V；target layer 只投影自己的 Q，并消费提前生成的 K/V。

以当前 48 层 WeLM 为例，基础 mirror 关系为：

```text
source layer 15..1  -> target layer 33..47
```

`source layer 0 -> target layer 48` 属于 NextN/MTP。基础 Deferred 只截断主模型的
mirror suffix；启用 WeLM MTP 时，Prefill 还会把 NextN 所需的 Draft prefix KV
直接写入 Draft KV pool，供 Decode 侧后续 verify/draft 使用。

Mirror 结构使 target layer 的 prefix K/V 可以在 source layer 提前产生。
Deferred 模式进一步在 source 侧完成目标层要求的 K normalization、位置处理、
RoPE 和 KV pool 路由，并直接写入 target layer 的 KV cache。因此，初始
Prefill 不必为了保存 target KV 而继续执行整个 mirror suffix。

### Prefix、seed 与 first output

对于长度为 `N` 的 prompt：

```text
M                 = N - 1
committed prefix  = prompt[0:M]
seed              = prompt[M]
```

- `committed prefix` 由 Deferred Prefill 处理并写入 KV cache。
- `seed` 是最后一个 prompt token，它不是生成结果。
- seed 在普通 Decode 路径中执行后，采样得到第一个用户可见输出 `y0`。

这种拆分保留了自回归语义：seed 仍位于原始位置 `M`，能够读取完整 prefix KV；
`y0` 仍是模型在完整 prompt 之后产生的第一个输出 token。

## 为什么需要 Deferred

Prefill 的主要优势来自长序列上的大规模并行计算。其 attention、线性层和 MoE
算子通常围绕较大的 token 维度组织，以提高 GPU 利用率。

Mirror suffix 的计算形态与之不同。Source layer 已经生成了 target K/V，而
mirror 优化会把 suffix 的有效 Q 行收缩到每个请求的最后一个位置。结果是：

- 前半段是长序列、计算密集的 Prefill；
- 后半段变成每个请求少量 Q 行的 mirror suffix；
- 这段小行数计算仍占据 Prefill 执行链，并附带 kernel launch、同步、最终
  norm、LM head 和采样等开销；
- Prefill 算子和 batch 组织并不是这种小行数 workload 的最佳执行环境。

Decode 本身就是“一条请求一行 token”的执行模式。将 seed 放入 Decode 后，
它可以与其他 READY seed 或普通 Decode 请求叠 batch，并复用 Decode CUDA
Graph、paged KV metadata 和面向小行数优化的 attention/线性层路径。对于 P/D
分离部署，Prefill worker 还可以在 Decode worker 执行 seed 时继续接收新的
Prefill 工作，从而把两部分负载分配到更合适的设备上。

## 执行流程

Deferred 模式把一次初始推理拆成两个连续阶段：

```text
prompt = [t0, t1, ..., tM-1, tM]

Deferred Prefill
  input:  [t0, ..., tM-1]
  run:    base layers 0..32
  write:  ordinary KV + finalized target mirror KV
  skip:   layers 33..47, final norm, LM head and sampling
  result: typed Prefill completion
                         |
                         v
Seed Decode
  input:  tM at position M
  run:    complete model through the ordinary Decode path
  result: first generated token y0
                         |
                         v
Ordinary Decode
  input:  y0, y1, ...
```

Deferred Prefill 返回的是类型明确的 `WelmDeferredPrefillCompletion`，不是
占位 logits，也不是伪造的生成 token。这样可以避免 seed 被提前计入
`output_ids`、grammar、penalty、streaming 或生成 token 指标。

启用 WeLM MTP 时，Prefill 同样只提交完整 prefix 的 Target/Draft KV，不返回
bootstrap hidden state。seed 进入普通 MTP target verify batch，但首轮只消费
canonical root token：该轮 `accept_len=1`，不会把尚未由完整 prompt 验证的 draft
token 计入输出。首轮 continuation 完成后，请求恢复普通 MTP 生命周期。READY seed
可以和普通 MTP verify rows 叠 batch；实现允许重算 seed 对应的单个 token，以保持
原有 verify/continuation 数据流。

单体部署仍只加载一份完整模型。运行 Deferred Prefill 时通过 batch metadata
动态设置执行截止层；运行 seed 和后续 Decode 时则执行完整模型。P/D 分离部署中，
Prefill worker 不加载完整 MTP Draft 执行权重，只构造写入 Draft prefix KV 所需的
storage-only Draft KV pool；Decode worker 保持完整 Target/Draft 执行能力。

## 新增的调度机制

### 统一的 seed 生命周期

每个请求使用一个显式状态机：

```text
PREFILL_PENDING -> READY -> INFLIGHT -> CONSUMED
```

- `PREFILL_PENDING`：committed prefix 尚不能被 seed Decode 使用。
- `READY`：prefix KV 和 request-table 映射已经提交，seed 可以进入 Decode。
- `INFLIGHT`：seed 的 KV slot 已分配，本轮正在执行 seed。
- `CONSUMED`：seed 已产生第一个真实输出，请求转入普通 Decode。

状态机同时覆盖单体和 P/D 分离部署。两者只在 `READY` 之前不同：单体模式等待
本地 Prefill/Radix 提交，P/D 模式等待 KV 传输和 Decode 侧提交。

### READY seed 合并到普通 Decode batch

Scheduler 在每轮调度开始时从 waiting queue 中选择 `READY` seed，并将它们构造
成标准 Decode rows，随后合并到已有的 `running_batch`。seed 使用普通
`prepare_for_decode()` 完成 KV 分配、sequence length 更新和采样状态构造。

该机制没有引入 seed 专用执行队列，也没有改变 SGLang 原有的 Prefill 优先级。
当本轮仍能生成新的 Prefill batch 时，Scheduler 仍优先运行 Prefill；当进入
Decode 时，READY seed 可以与其他 seed 或普通 Decode rows 共同执行。

### 与 overlap scheduler 协同

在 overlap 模式下，下一轮调度可能发生在上一轮结果处理之前。Deferred 路径
因此在 forward stream 上产生 typed completion，并在暴露 `READY` 之前完成：

1. committed prefix 的 Radix/request-table 发布；
2. seed token 写入下一轮输入；
3. `PREFILL_PENDING -> READY` 状态转换；
4. FutureMap 结果发布。

Radix 更新使用已有 schedule stream 顺序，不增加全局 device synchronize，也
不恢复会串行化 Prefill 的 process-before-schedule barrier。

### Radix Cache 与 page size

Deferred 仍使用原有 Radix Cache。对于 committed length `M` 和 page size `P`：

```text
R = floor(M / P) * P
```

Radix 最多公开复用前 `R` 个 token，非 page-aligned 的 `[R, M)` 保留为请求私有
tail。若 `M` 本身 page-aligned 且完全命中，Scheduler 可以直接把请求变为
`READY`，不执行 Prefill forward，也不分配 dummy page 或 dummy slot。

Abort、timeout、priority eviction 和 retraction 统一释放 request row、Radix
lock、私有 KV tail 与未提交的 seed slot。若单体 retraction 已释放 committed
KV，请求回到 `PREFILL_PENDING` 并重新匹配 Radix；若底层仍保留 prefix，则可以
从 `READY` 重试。

WeLM MTP 默认使用 direct-pool：NextN mirror K/V 在 Prefill 中直接写入对应 Draft
KV pool，不随 completion 携带中间 mirror tensor。该机制与是否启用 Deferred
解耦，P/D legacy 和 Deferred 调度都使用相同存储语义。P/D 传输会同时描述 Target
和 Draft KV pool，并要求两者 page size 与 allocator 映射一致。当前支持配置固定
为 `--page-size 16`。

### DP 协同

DP attention 下，不同 DP rank 可能同时运行 Deferred Prefill、普通 Prefill、
Decode 或 idle row。Scheduler 会同步每个 slot 的 Deferred 标记。在 cutoff
之后，Deferred slot 的本地 token 数收缩为零，但所有 rank 仍使用一致的全局
布局参与后续 MLP/EP collective，避免某个 rank 提前退出造成通信失配。

## 预期收益

Deferred Mirror Layer 的收益来自工作放置方式，而不是改变模型计算结果：

- **降低 Prefill 负载**：跳过初始 Prefill 中的 mirror suffix、最终 norm、
  LM head 和采样，将 Prefill worker 留给更适合的大规模 token 计算。
- **提高小行数计算的合批机会**：多个 seed 可进入同一个 Decode batch，也可与
  已有 Decode rows 合并，提高 mirror suffix 的有效 batch size。
- **复用更适合的算子路径**：seed 使用 Decode CUDA Graph、paged attention 和
  小行数线性层路径，避免在 Prefill 尾部运行零散的小 workload。
- **改善 P/D 负载分工**：P worker 更早结束请求的 Prefill 部分，D worker 在有
  batch 余量时吸收 seed，允许两侧并行推进。
- **保持单体部署的内存效率**：单体模式动态截止 Prefill，但只保存一份完整模型
  权重，不需要额外复制一个 Prefill 或 Decode 模型实例。
- **保留 prefix reuse**：page-aligned Radix prefix、私有尾页和 full-hit
  zero-forward 语义保持不变。

实际收益取决于 prompt/output 比例、mirror suffix 成本、Decode batch 填充率、
CUDA Graph bucket、Prefill 调度压力，以及 P/D 两侧是否有可重叠工作。Deferred
提供的是把计算移动到更合适执行阶段的机制，不保证所有 workload 都获得相同
幅度的吞吐或延迟提升。

## 当前边界

当前实现面向纯文本 `WeLMV4MoeForCausalLM`，并要求 Prefill/Decode 使用 FA3。
Speculative decoding 仅支持 WeLM NextN/MTP、Spec V2、`topk=1` 的线性路径；其他
speculative algorithm 和 tree verify 不在支持范围内。暂不支持 mixed chunk、
pipeline parallelism、HiCache、LoRA、suffix parallel、Scale-Seq、
previous-precision execution 或非 BF16 KV cache。当前 MTP runtime 尚未支持
AttnCP，因此不在验证矩阵中；Deferred 生命周期本身不依赖具体并行策略。P/D
分离模式当前使用 Mooncake 作为 KV transfer backend。

`SGLANG_WELM_MTP_LEGACY_MIRROR_STATE=1` 只保留用于 P/D legacy 模式的短期兼容
回退。它会恢复 completion 携带 mirror tensor 的旧路径，默认关闭、不会继续扩展，
并计划在 direct-pool 完成迁移后删除。

不满足 fast path 前置条件时会显式报错，不会静默回退到完整 Prefill 或其他低性能
路径。
