# 3. Phase-2 finetuning/training — design research + LayerNorm backward fix

- Status: in progress (backward prerequisite done, scope decided, loss/matching
  implemented and validated, end-to-end demo working with a known synthetic-input
  numerical caveat, dataloader/optimizer-loop not yet implemented)
- Date: 2026-07-18

## Context and Problem Statement

All three inference milestones (detection, segmentation, keypoints) are
done for the Nano-family variants plus RFDETRBase. The project's original
scope defers "phase 2" (finetuning/training support) until inference is
solid — that condition is now met. Per explicit instruction: research and
design this phase properly before writing any training code (unlike every
inference milestone so far, there's no precedent in the sibling ports —
`see-through-cpp` and `trellis2cpp` are both inference-only). What would a
C++/ggml training loop for this codebase actually require?

## Finding 1: ggml has a real training/autodiff framework — this isn't from scratch

Contrary to the "no precedent" framing above at the _port_ level, `ggml`
itself ships genuine training infrastructure, not just inference primitives:

- `ggml_set_param()` marks trainable tensors; `ggml_build_backward_expand()`
  builds the gradient graph from a forward graph.
- `ggml_opt_step_adamw()` / `ggml_opt_step_sgd()` — optimizer step ops,
  operate as ordinary graph nodes (so they run through the same
  CPU/Vulkan backend dispatch as everything else).
- `ggml-opt.h` — a higher-level dataset/context/epoch API
  (`ggml_opt_dataset_*`, `ggml_opt_context_t`, `ggml_opt_fit`,
  `ggml_opt_epoch`) for a standard supervised training loop, with a
  progress-bar callback already provided.
- `ggml_cross_entropy_loss()` / `_back()` exist as first-class ops.

This means the _mechanics_ of a training loop (graph-based backward pass,
optimizer step, epoch/batch iteration) are provided by ggml — this port
doesn't need to hand-roll autodiff or an optimizer.

## Finding 2: this port's _existing_ graphs would crash `ggml_build_backward_expand` today

Checked ggml's backward-pass switch (`ggml_compute_backward` in
`third_party/ggml/src/ggml.c`) against every op this codebase actually
uses. **The `default` case is `GGML_ABORT("unsupported ggml op for backward
pass")`** — not a silent no-op, not a warning, a hard process abort. Ops
with **no backward case at all**, cross-referenced against where this port
uses them:

| Op (no backward)                                 | Used where in this port                                                                                                          | Impact                                                                                                                                                                                                                                                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GGML_OP_NORM` (plain LayerNorm via `ggml_norm`) | `layer_norm_affine`, `spatial_layer_norm_affine` (`ops.h`) — used in **every** block: backbone, decoder, segmentation, keypoints | **Blocking.** This is the single most-used normalization op in the whole port. `GGML_OP_RMS_NORM` _does_ have backward (`ggml_rms_norm_back`, a dedicated op) — LayerNorm's is a well-defined, structurally similar formula, just not implemented upstream yet.                 |
| `GGML_OP_TOP_K`                                  | `rfdetr_decoder`'s two-stage query selection                                                                                     | Not blocking in a useful sense — top-k _selection_ is inherently non-differentiable (a discrete index choice); the reference DETR implementation doesn't backprop through it either, so this needs isolating from the trainable subgraph, not a ggml patch.                     |
| `GGML_OP_POOL_1D`                                | decoder's max-over-classes reduction, segmentation's would-be use (not currently used for anything trainable-adjacent)           | Same as top-k — used for score ranking, not a differentiable path in the reference implementation either.                                                                                                                                                                       |
| `GGML_OP_CLAMP`                                  | `deform_attn.cpp`'s bilinear sampling (index clamping)                                                                           | Expected: index computations aren't meant to be differentiable _through the index itself_, but the deformable-attention _sampling weights_ (bilinear interpolation coefficients) mathematically **do** have a well-defined gradient w.r.t. the learned offsets — see Finding 3. |
| `GGML_OP_UPSCALE`/interpolate (bicubic/bilinear) | segmentation's spatial upsample, backbone's position-embed resize                                                                | The position-embed resize isn't trained (it's a fixed conversion-time-baked matrix, see `0002-position-embed-bicubic.md`) — not blocking. Segmentation's upsample of _learned_ features would need backward if segmentation is finetuned.                                       |

Ops confirmed **to have** backward support and used extensively here:
`MUL_MAT` (every `linear()`), `GET_ROWS` (deformable attention's corner
gather, embedding lookups), `IM2COL` (`conv2d`'s convolution), `SOFT_MAX`
(every attention block), `ADD`/`MUL`/`SUB`/elementwise, `RESHAPE`/`VIEW`/
`PERMUTE`/`CONT` (pure bookkeeping, always differentiable). This is most of
the FLOPs in the network — the gap is narrower than "nothing works," but
`GGML_OP_NORM`'s absence alone blocks naive end-to-end backprop through
literally every existing block.

## Finding 3: deformable attention needs a _hand-derived_ backward, not just missing-op patches — RESOLVED

`src/deform_attn.cpp`'s bilinear sampling is built from `ggml_floor` +
`ggml_clamp` + `ggml_get_rows` + elementwise weight multiplies — a
decomposition chosen because ggml has no native `grid_sample`-equivalent
(see `docs/decisions/decoder.md`). Even if every individual op in that
chain had a backward case, composing them via autodiff would differentiate
through `ggml_floor`'s corner-index computation, which is not how
deformable attention's real gradient is defined (the _indices_ aren't
differentiable; the _bilinear weights_ are, w.r.t. the continuous sampling
location which itself depends on the learned `sampling_offsets`).

**Resolution (later session, once the decoder itself needed to be
trainable): no custom op needed.** `floor` and `clamp` already exist as
ops with working forward kernels in ggml — they were only missing from
`ggml_compute_backward`'s dispatch switch in `third_party/ggml/src/ggml.c`.
Two small additions, both following exact patterns already present in that
file:

- `GGML_UNARY_OP_FLOOR`: a "noop" (zero) backward case, identical to the
  treatment ggml already gives `SGN`/`STEP` (both piecewise-constant, true
  derivative zero almost everywhere). This is what makes
  `frac(x) = x - floor(x)` autodiff to the mathematically correct
  `d/dx = 1`: `SUB`'s own backward sends `+grad` to `x` and (this noop)
  `+0` to `floor(x)` — exactly the "indices aren't differentiable, the
  bilinear weight is" distinction this Finding originally called for,
  achieved without any special-casing in `deform_attn.cpp` itself.
- `GGML_OP_CLAMP`: a real gate gradient (1 inside `[lo,hi]`, 0 outside),
  the same step-function shape as `RELU`'s existing backward (which gates
  on one boundary instead of two).

A third, related gap was found while validating: `ms_deform_attn`'s
per-head output assembly used `ggml_concat` (no backward case at all, same
class of gap) — fixed the same way `bbox_reparam_decode_diff` already
fixed the identical problem for box decoding: assemble via `ggml_set`
(writing each head's disjoint channel slice into a scratch tensor) instead
of `ggml_concat`. Forward-identical, zero regression on the full test
suite.

Validated with `tests/test_deform_attn_backward.cpp`: finite-differences
the gradient of a real `ms_deform_attn` call w.r.t. both `query` (drives
`sampling_offsets`, threading through the floor/clamp path) and
`ref_points` (feeds the sampling location via two separate views) —
matches to ~1e-4-level relative error across many indices, for the full
`n_heads=2, n_points=2` case.

**A separate, genuine ggml bug found during validation** (not a
correctness bug — it doesn't affect computed values): freeing
(`ggml_free`) the graph context after a backward pass through
`ref_points` (but not through `query`) segfaults, even though the
gradient values are already read out and verified correct _before_ the
crash. Extensively bisected: a from-scratch minimal repro of the exact
same floor/clamp/get_rows/multi-view/`ggml_set` computation graph does
**not** crash on free; only the real `ms_deform_attn`-produced graph does,
and only for the `ref_points`-as-param case specifically. This is a
memory-management bug in ggml's own context-teardown bookkeeping for this
graph topology, not a gradient-correctness issue. Workaround (matching
this project's established pattern for other real ggml gaps): the
backward-graph context is deliberately **leaked** (never `ggml_free`'d)
rather than crash — acceptable for a bounded test; a long-running
trainable-decoder loop would need the same treatment (or a proper ggml-side
fix, not attempted here) for every step that backprops through
`ms_deform_attn`.

## RF-DETR's loss requirements (from `rfdetr` source, read this session)

- **Hungarian matching** (`scipy.optimize.linear_sum_assignment` in
  upstream, or an equivalent): a **non-differentiable, discrete assignment**
  between predicted queries and ground-truth targets, computed _outside_
  the gradient graph as ordinary CPU preprocessing on detached tensors —
  this is standard for every DETR-family model and is not a ggml-autodiff
  concern at all; it just needs a plain C++ (or even a small Python
  preprocessing step feeding fixed assignment indices into the ggml graph)
  implementation, decoupled from backward-pass concerns.
- **Detection loss**: focal loss (classification) + L1 + GIoU (box
  regression), applied only to matched pairs. Focal/L1 are elementwise —
  fine once `NORM` backward exists upstream of them. GIoU needs its own
  elementwise formula (intersection/union of predicted vs. target boxes) —
  not yet checked whether this decomposes into already-backward-capable
  ops; likely yes (min/max/mul/div/add are all in the supported set) but
  unverified.
- **Segmentation loss**: (not yet read in detail — deferred, this
  research pass focused on the backward-op-support blocker first since
  it's a hard prerequisite for _any_ task's training, not task-specific).
- **Keypoint loss**: partially read this session
  (`compute_l1_keypoint_loss`, `docs/decisions/keypoints.md`'s earlier
  research) — an L1 term + BCE (findable/visible) + a Gaussian NLL with
  Cholesky-parameterized precision. All elementwise/reduction ops, same
  "blocked only by NORM" caveat as detection.

## Prerequisite resolved: `layer_norm_affine_diff` (`src/ops.{h,cpp}`)

Rather than patching vendored ggml (adding a new `ggml_norm_back` op), the
LayerNorm-backward gap from Finding 2 is resolved by decomposing
`ggml_norm` into primitive ops that each individually have a backward case.
This turned out to need more care than "pick primitives with a `case` in
`ggml_compute_backward`'s switch" — two additional, non-obvious shape
constraints surfaced during validation (`tests/test_norm_backward.cpp`):

- **`ggml_sub`/`ggml_div`'s backward doesn't broadcast.** `GGML_OP_ADD` and
  `GGML_OP_MUL`'s backward call `ggml_repeat_back` on the gradient before
  writing it to a smaller (broadcast) `src1`; `GGML_OP_SUB`/`GGML_OP_DIV`
  don't — they assume `src1` is already the output shape and hit
  `GGML_ASSERT(ggml_are_same_shape(src1, cgraph->grads[isrc1]))`. Centering
  (`x - mean`, `mean` is `(1,T,N)` broadcasting against `x`'s `(C,T,N)`)
  and normalizing (`centered / std`) both broadcast, so both are rewritten
  through the add/mul path instead: `add(x, scale(mean, -1))` and
  `mul(centered, reciprocal)` where `reciprocal = div(ones, std)` (that div
  is same-shape, non-broadcast — safe).
- **`ggml_mean`'s own backward assumes a true scalar output.** It's
  implemented via `ggml_add1_or_set`, which asserts its operand is a scalar
  (`ggml_is_scalar`) — fine for the "mean of a whole tensor down to one
  number" case ggml's own code uses it for, but `layer_norm_affine_diff`
  needs a per-row mean (output `(1,T,N)`, not `(1,1,1,1)`). `ggml_sum_rows`
  has the identical forward reduction shape but a shape-general backward
  (plain `ggml_repeat`, no scalar assumption), so mean is computed as
  `scale(sum_rows(x), 1/C)` instead of `ggml_mean(x)`.

Validated in `tests/test_norm_backward.cpp`: forward output matches
`layer_norm_affine` (fused `ggml_norm`) to 7.2e-7; `ggml_build_backward_expand`
on a graph using `layer_norm_affine_diff` completes without aborting; the
resulting gradient matches a central-difference numerical check to <0.1%
relative error on 5 sampled elements. The existing fused `layer_norm_affine`
is untouched (still used by every inference graph) — `layer_norm_affine_diff`
is a separate, opt-in function for graphs that need backward.

## Scope decision: detection-head-only finetune (superseded — see below)

**UPDATE (later session): scope widened.** The user explicitly requested
full DETR (decoder) finetuning, not just the detection heads — this
directly motivated resolving Finding 3 (deformable-attention backward),
now done (see the updated Finding 3 section above). The original
head-only scope decision below is kept for the historical record of _why_
it was chosen at the time (it was the smallest useful slice, and it
sidestepped exactly the two things — `ms_deform_attn` backward and
`layer_norm_affine_diff` wiring — that a wider scope needs).

**Originally chosen scope: freeze the DINOv2 backbone AND the transformer
decoder; finetune only `class_embed`/`bbox_embed`** (the final linear/MLP
heads reading the last decoder layer's hidden state). This is the smallest
useful slice flagged in the prior draft of this doc, and it's the one that
sidesteps the most open work:

- Freezing the decoder means `ms_deform_attn`'s backward (Finding 3, a
  hand-derived analytical gradient, not a small task) is **not needed** —
  gradients never need to flow through it since nothing upstream of
  `class_embed`/`bbox_embed` requires a gradient.
- Freezing the backbone means the windowed-attention machinery
  (`dinov2_backbone`) also never needs backward support.
- `class_embed`/`bbox_embed` are plain `linear()`/`mlp()` calls — already
  fully backward-capable (`MUL_MAT`, `ADD` both have backward cases) with
  **zero new primitive work** beyond what's already validated.
- This scope still exercises the real prerequisite this phase closed:
  `class_embed`/`bbox_embed`'s INPUT (the decoder's final `hidden_states`)
  is itself the output of a `layer_norm_affine`-heavy stack, but since that
  stack is frozen (no gradient requested through it), `layer_norm_affine`
  (the fast, backward-incapable fused version) can stay as-is there too —
  `layer_norm_affine_diff` isn't even needed for THIS scope specifically,
  though it remains available (and validated) for whenever the scope
  widens to include the decoder.

Remaining steps for this scope, in order:

1. **Done.** Loss + Hungarian matching (`src/loss.{h,cpp}`): host-side
   Hungarian matching (a standard O(n²·m) Kuhn-Munkres, matching upstream's
   exact `matcher.py` cost formula — checkpoint-verified against
   `scipy.optimize.linear_sum_assignment`'s own real output, not assumed)

   - a ggml loss graph (sigmoid focal loss + L1 + GIoU). Deliberately
     reproduces upstream's `ia_bce_loss=False` (plain `sigmoid_focal_loss`)
     path, NOT the actual default `ia_bce_loss=True` variant — see the
     "IA-BCE not implemented" note below for why. `ggml_sigmoid` has no
     backward case either (same class of gap as `ggml_norm` — see Finding 2)
     so classification probability is computed via the same
     primitive-decomposition trick as `layer_norm_affine_diff`:
     `sigmoid(x) = 1/(1+exp(-x))` from SCALE/EXP/ADD/DIV (all backward-
     capable). GIoU's elementwise min/max (no native ggml op) is built from
     `a + relu(b-a)` / `a - relu(a-b)`, both backward-capable.

   Validated in isolation (`tests/test_loss.cpp`) against the real
   upstream `HungarianMatcher`+`SetCriterion` on synthetic data: matched
   (query,target) pairs match exactly (discrete assignment, no tolerance),
   the scalar loss value matches to 2e-6, and a finite-difference check on
   both `pred_logits`/`pred_boxes` gradients confirms the graph is
   genuinely differentiable and correct. One real bug caught while
   building the reference: the upstream formula
   `sigmoid_focal_loss(...).mean(1).sum()/num_boxes * num_queries` looks
   like it needs a `1/num_classes` scale (`.mean(1)` over what looks like
   a class axis) but `.mean(1)` is actually over the QUERIES axis
   (`src_logits` is `(batch,queries,classes)`), and that `1/num_queries`
   exactly cancels the outer `*num_queries` — the correct combined formula
   is simply `sum-over-everything / num_boxes`, no `num_classes` term at
   all. An initial wrong guess (dividing by `num_classes` AND keeping the
   `num_queries` factor) produced a ~2.2x-too-large loss until caught by
   diffing against the real reference value.

   **IA-BCE not implemented** (documented gap, not a bug): upstream's
   actual default (`ia_bce_loss=True`) weights each positive example by
   `prob^alpha * iou^(1-alpha)` where `iou` is computed from
   `pred_boxes.detach()` — i.e. a value that must be computed from the
   CURRENT forward pass but explicitly excluded from the backward graph
   (a "stop-gradient"). ggml has no direct stop-gradient primitive; doing
   this correctly needs a two-pass approach (compute IoU in a separate
   forward-only sub-graph, read its VALUE back to the host, feed it back
   in as a plain host constant for the main loss graph) — deferred as a
   documented follow-up, not attempted this pass. The plain
   `sigmoid_focal_loss` path chosen instead is a real, upstream-provided
   alternative (not something invented for this port), just not the
   default.

2. **Done.** End-to-end training-step demo
   (`demos/train_step_demo.cpp`): loads the real RFDETRNano checkpoint,
   runs the full frozen backbone→projector→decoder forward pass, matches
   against a REAL COCO image's real annotations via `hungarian_match`
   (`gen_reference/gen_reference_real_image.py`, upstream's own
   `val_speed` transform pipeline on `000000289343.jpg`: person, dog,
   bicycle, bench), builds `detection_loss`, backprops, and applies a
   manual AdamW update (same formula/parameter layout as
   `ggml_opt_step_adamw`'s own CPU kernel, hand-rolled rather than using
   the graph op directly — see the note below) to `class_embed`/
   `bbox_embed` only. Proves the FULL scope-decided plumbing genuinely
   composes: weight-shadowing a `Model`'s frozen GGUF tensors with fresh
   per-step trainable ones (via `m.weights[name] = t`), two fresh graphs
   per step (forward-only for matching, then forward+loss+backward+
   AdamW), carrying updated weights host-side between steps. Loss
   decreases steadily and stays bounded over 30 steps (0.59 → ~0.10 at
   lr=0.01), genuinely fitting the image's 4 real annotations.

   Four more real bugs found and fixed while building this (beyond the
   `mean(1)` and `ggml_sigmoid` ones already listed above):

   - **`ggml_concat` has no backward case** (same gap class as
     `ggml_norm`/`ggml_sigmoid`) — `bbox_reparam_decode`'s final
     `ggml_concat(cxcy, wh, 0)` assembly of `pred_boxes` aborted the
     process the instant `bbox_embed`'s output needed a gradient through
     it. Fixed with `bbox_reparam_decode_diff` (`src/decoder.cpp`), which
     assembles the same result via two `ggml_set` calls instead (SET _is_
     backward-capable) — forward-identical, confirmed by swapping it in
     unconditionally at the one call site that builds `pred_boxes` and
     re-running all 7 existing decoder/segmentation/keypoint inference
     tests with zero regression.
   - **MSVC Debug-build assertion failures pop a blocking interactive
     dialog** instead of exiting — fatal for a headless run (stdin is
     `/dev/null`, so the dialog just hangs forever; this is almost
     certainly what an earlier session's "hung modal" report was too,
     from the antialias+bicubic assert). Fixed by calling
     `_CrtSetReportMode`/`_CrtSetReportFile` (routing asserts to stderr)
     and `_set_abort_behavior` (disabling the abort-message-box and
     Windows Error Reporting) at the top of `main()`, guarded by
     `#if defined(_MSC_VER) || defined(_WIN32)`.
   - **The demo's own missing `output_proposals` upload** — this is what
     was actually behind the earlier-suspected "synthetic random-noise
     input" numerical blowup, and it reproduced identically with real
     COCO data too, disproving that hypothesis. `rfdetr_decoder`'s
     two-stage proposal heads read `output_proposals`
     (`src/decoder.h`/`.cpp`) as a required INPUT tensor — the demo built
     the graph but never called `ggml_backend_tensor_set` on it, so the
     encoder's per-location candidate boxes were uninitialized allocator
     garbage. `wh = exp(clamp(delta,-4,4)) * base_wh` bounds `delta` but
     not `base_wh`, so a garbage `base_wh` (observed up to ~3e7) produced
     `pred_boxes` up to ~6.2e10 and a loss around -5e26. Fixed by calling
     `output_proposals_data(gw, gh)` (already existed, just unused by this
     demo) and uploading it in `build_graph`.
   - **Trainable param tensors weren't protected from graph-allocator
     buffer reuse** — `class_embed`/`bbox_embed`'s fresh per-step tensors
     were `ggml_set_input`+`ggml_set_param`'d but not `ggml_set_output`'d.
     Once each param's last forward-graph consumer ran, the allocator was
     free to hand its buffer to a _later_ node's output — observed
     directly: `head.class_w[0]` (read back after `ggml_backend_tensor_
get`) came back bit-identical to that step's loss scalar, and this
     happened even with `lr=0` (a true no-op update), proving loss values
     were leaking into weight buffers rather than the weights actually
     drifting from training. This is the same allocator-buffer-reuse bug
     class already found and worked around for `g.loss` itself (see
     `ggml_set_output(g.loss)`'s comment) and for the SegXLarge decoder
     graph and `test_loss.cpp`'s repeated-execution case — evidently the
     fix needs applying to _every_ tensor read back after compute, not
     just the final loss. Fixed by adding `ggml_set_output(t)` in
     `make_head_tensor` for all 8 trainable tensors.

3. Dataset/dataloader beyond this one hardcoded image (COCO-format
   annotations → ggml input tensors for an arbitrary image, batching,
   iteration over a full split) — not researched yet. The single-image
   real-data path (`gen_reference/gen_reference_real_image.py`) proves the
   preprocessing/annotation pipeline is correct but isn't a general
   dataloader.
4. `ggml_opt_step_adamw`/`ggml_opt_epoch` wiring — the demo currently
   hand-rolls the identical AdamW math as a host-side loop rather than
   using the real graph op, specifically to sidestep the "second real ggml
   training-infra gap" below (a graph op would need to be part of the
   SAME single-compute graph as forward+backward, adding complexity for
   uncertain benefit given that gap's root cause isn't understood yet).
   Worth revisiting once that's root-caused.

## A second real ggml training-infra gap found: repeated graph execution

While validating `detection_loss`'s backward pass, re-running the SAME
allocated backward-enabled graph (`ggml_graph_reset` + `ggml_backend_
graph_compute`, in a loop, for finite-difference perturbation checks —
the exact pattern `tests/test_norm_backward.cpp` already used successfully
for the simpler LayerNorm case) silently corrupted the result: a plain
_re-evaluation at the same unperturbed point_ returned 67.9 instead of the
correct 23.6 the second time the graph was computed. Rebuilding a fresh
context/graph/allocator for every single evaluation (discarding graph
reuse entirely) fixed it. Root cause not tracked down — likely related to
the SegXLarge decoder graph-allocator bug documented separately in
`docs/decisions/0001-open-work.md`, since both involve a graph being
computed more than once (there: multiple downstream consumers within one
compute; here: multiple separate `ggml_backend_graph_compute` calls on one
allocated graph) producing a corrupted value despite `ggml_graph_reset`.
Flagging this as a second data point for whoever eventually debugs the
underlying `ggml-alloc.c` issue — it may be the same root cause showing up
in two different reuse patterns.

Widening the scope later (decoder trainable, then backbone trainable) is a
separate future decision — each widening reuses `layer_norm_affine_diff`
directly (already validated) except the decoder-trainable step, which
additionally needs Finding 3.

## Consequences

This doc intentionally stops short of a full training loop, per the
explicit instruction not to rush Phase B implementation. The load-bearing
finding is Finding 2: **training was blocked on a real ggml gap
(`GGML_OP_NORM` backward), not on this port's own code** — that gap is now
closed (`layer_norm_affine_diff`, validated), which unblocks every
finetuning scope from "just the classifier head" to "everything" (all of
them route through at least one LayerNorm). The remaining prerequisite
before any RF-DETR-specific training-loop code is Finding 3's deformable-
attention backward, and only if the chosen scope needs the decoder
trainable.

## Parked

The avatar fine-tune path (`pixi run -e train`, `scripts/coco_person_subset.py`,
`scripts/finetune_seg.py`) is built and proven; the run itself is parked.

- Shelved 2026-09-27: the full RFDETRSegNano avatar fine-tune (about 20
  epochs on VRM renders plus pseudo-labels plus CC-BY COCO persons). The
  loop works: a 1-epoch dry run on an RTX 4090 ran at 365 ms/iter median
  with a 4.6 GiB peak, and all four `convert_*_to_gguf.py` converters and
  `seg_cli` load its output. Not run because no vehicle on RFD 2262's board
  needs avatar-person detection; faces are the Car. Unpark when one does.
- Shelved 2026-09-27: publishing fine-tuned weights to Hugging Face
  (`chibifire/...`). Nothing to publish yet. Unpark with the fine-tune
  above.
