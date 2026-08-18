# WeLM Token-Owner DeepEP MTP CUDA Graph Design

This document extends
[`token_owner_design.md`](token_owner_design.md).
All owner-layout and KV-mirror invariants from the main design remain in force;
this document only specifies the DeepEP/MTP CUDA Graph delta.

## 1. Goal

Extend the existing WeLM token-owner path so decode-side DeepEP, MTP/NextN,
and CUDA Graph can run together. The same ranks must also support non-separated
prefill and decode serving without restarting the process or changing the
token-owner layout.

This phase uses the existing `--enable-token-owner` and DeepEP configuration.
It adds no environment variable, communication primitive, owner definition, or
KV-cache transfer path.

## 2. Supported Configuration

The target model and MTP draft model both use DeepEP. An unset
`--speculative-moe-a2a-backend` inherits the target DeepEP backend.

Two DeepEP modes are supported:

| `--deepep-mode` | Deployment | Prefill/extend | Decode target verify | MTP proposal |
|---|---|---|---|---|
| `auto` | non-separated or decode-only | normal eager path | low-latency CUDA Graph | low-latency CUDA Graph |
| `low_latency` | decode-only | not applicable | low-latency CUDA Graph | low-latency CUDA Graph |

`auto` is required for non-separated prefill/decode serving. Explicit
`low_latency` remains supported for decode-only disaggregated serving. It is
rejected for a process that can execute prefill because DeepEP low-latency
buffers are sized for decode rows rather than long prefill token batches.

The following combinations remain unsupported and must fail explicitly:

- `deepep-mode=normal` with this CUDA Graph path;
- `deepep-mode=low_latency` outside a decode-only deployment;
- target DeepEP with a non-DeepEP speculative MoE A2A backend;
- existing token-owner unsupported topologies or features.

## 3. Design

Use the existing DeepEP CUDA Graph adapter and the current token-owner metadata.
Do not create another dispatcher, graph runner, or communication path.

At startup, capability validation accepts speculative token-owner DeepEP only
when the effective draft backend is also DeepEP and the mode is `auto` or
`low_latency`. Invalid combinations fail before graph capture.

Target verify continues to use the existing model CUDA Graph runner, whose
DeepEP adapter captures decode with low-latency dispatch.

The WeLM MTP proposal graph records that its token-owner draft batch uses
DeepEP. During capture and replay it selects low-latency dispatch. KV-mirror
contraction may retain `ForwardMode.DRAFT_EXTEND` and SUM_LEN metadata, but it
must not change the DeepEP dispatch mode back to normal. A graph-local marker
only controls the `set_is_extend_in_batch` value consumed by DeepEP; it does not
change flattened MTP rows, owner counts, residual placement, KV writes, or
mirror contraction.

For `auto`, ordinary prefill and mixed-extend setup continues to publish extend
mode and therefore resolves to normal DeepEP. Decode graph replay restores its
captured low-latency mode on every replay. This permits the same process to run
normal prefill, low-latency decode graph, and normal prefill again. A
non-separated process configured with explicit `low_latency` fails during
capability validation instead of reaching a request-time DeepEP capacity
assertion.

## 4. Invariants

- Before KV-mirror contraction, token ownership uses the existing balanced
  flattened-row layout. After contraction, survivor rows remain on their
  original owners and the per-lane row counts may be uneven or zero.
- Target verify and draft proposal use the same logical request buckets and
  owner metadata as the Global-TP MTP Graph path.
- DeepEP collectives remain restricted to their existing EP group and ordering.
- Idle ranks execute the same graph/collective sequence with zero valid rows.
- Padding and non-survivor rows may participate in fixed-shape Router/TopK
  computation, but they are masked before DeepEP dispatch and cannot affect
  valid expert outputs, logits, logprob, or sampling results.
- No silent eager, Global-TP, or replicated fallback is allowed after selecting
  this fast path.
- Non-token-owner, Global-TP token-owner, pure TP, prefill CP, and KV-transfer
  behavior remain unchanged.

## 5. Validation

Focused tests must cover:

- capability acceptance for `auto` and `low_latency` with inherited or explicit
  DeepEP draft backend;
- rejection of `normal` and mismatched target/draft backends;
- proposal graph capture and replay selecting low-latency DeepEP;
- KV-mirror contraction preserving low-latency dispatch while retaining draft
  extend and owner metadata;
- mode transitions `normal prefill -> low-latency graph decode -> normal prefill`;
- idle rank, uneven batch, cache-hit, and mirror-contraction metadata;
- unchanged Global-TP MTP Graph and pure-TP behavior.

Remote ETE uses non-separated DP Attention + AttnTP + DeepEP serving with MTP
and CUDA Graph enabled. Logs must confirm target and proposal graph capture and
replay. Requests must complete without hangs, empty outputs, or fast-path
fallbacks across single-request, uneven-concurrency, and prefill/decode
transition cases.
