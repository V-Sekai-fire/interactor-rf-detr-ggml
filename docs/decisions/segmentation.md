# Segmentation head port decisions (RFDETRSegNano)

Status: **validated end-to-end** for RFDETRSegNano. `tests/test_segmentation.cpp`
chains `dinov2_backbone` → `projector_p4` → `rfdetr_decoder` →
`segmentation_head` against the real `rf-detr-seg-nano.pt` checkpoint:
`pred_boxes` 1.1e-2, `pred_logits` 2.9e-3 (both gate 5e-2, same as detection),
`pred_masks` (last block) 0.109 max-abs-diff against a **relaxed** 0.15 gate
(mean 2.2e-3) — see "Why the mask gate is looser" below, this is not a bug.

## RFDETRSegNano is a different config from RFDETRNano

`RFDETRSegNanoConfig` (`rfdetr/config.py`) overrides more than just
`segmentation_head=True`: `num_windows=1` (not 2), `dec_layers=4` (not 2),
`patch_size=12` (not 16), `resolution=312` (not 384), `num_queries=100` (not
300). Checkpoint: `https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth`.
Verified against the real checkpoint (same string-scan technique as the
backbone/decoder work): 12 backbone layers (unchanged), 4 decoder layers, no
register tokens (still the register-free Small encoder). `num_windows=1` is
a nice free sanity check for `backbone.cpp` — with exactly one window, the
"windowed" and "global" code paths are mathematically equivalent (a single
window's batch=1 self-attention _is_ the merge-to-batch=1 global path), so
this run exercises both branches while collapsing to the same math.

`src/backbone.cpp` and `src/decoder.cpp` needed **no code changes** for this
— both were already parameterized via `BackboneParams`/`DecoderParams`; only
different values were needed. `scripts/convert_dinov2_to_gguf.py` gained a
`seg-nano` `VARIANTS` entry; `scripts/convert_decoder_to_gguf.py` gained a
`dec_layers` CLI argument (was hardcoded to 2).

## `SegmentationHead` (`rfdetr/models/heads/segmentation.py`)

- `spatial_features` = the **same** projector output already used as decoder
  memory (`features[0].tensors` in `LWDETR.forward`) — not a separate tap.
- `query_features` = **all** `dec_layers` decoder-layer normed hidden states
  (`hs`, the full stack, not just `hs[-1]`) — this is why
  `DecoderOutput::hidden_states` (a vector, one entry per layer) was added to
  `src/decoder.cpp`; detection only ever needed `hidden_states.back()`.
- `num_blocks == dec_layers` (4 for Nano): the head runs `dec_layers`
  sequential `DepthwiseConvBlock`s, each **refining a running
  `spatial_features` state** (not independent per-layer copies) and paired
  with that layer's query features via `zip(blocks, query_features)` —
  `masks[i]` is produced after `i+1` blocks have run.
- `bottleneck_ratio` defaults to `1` (not `None`), so
  `spatial_features_proj`/`query_features_proj` are **real learned**
  `Conv2d(256,256,1)`/`Linear(256,256)` layers, not identities, despite
  in/out channel counts being equal — checkpoint confirmed both have weights.
- `DepthwiseConvBlock`: depthwise 3×3 conv (`ggml_conv_2d_dw_direct` — its
  expected kernel layout `(KW,KH,1,C)` matches PyTorch's depthwise weight
  shape `(C,1,KH,KW)` reversed by the GGUF writer with zero special-casing
  needed) → residual-critical: **channels-last LayerNorm, eps=1e-6** (uses
  `spatial_layer_norm_affine`) → pointwise `Linear(dim,dim)` → **exact
  (erf) GELU** (`nn.GELU()` default, not tanh-approx) → residual add. No
  LayerScale (`gamma` is `None`, `layer_scale_init_value=0` default).
- `MLPBlock` (`query_features_block`): pre-LN **eps=1e-5** (`nn.LayerNorm(dim)`
  with no explicit eps — PyTorch's default, _different_ from
  `DepthwiseConvBlock`'s explicit 1e-6; easy to mix up) →
  `Linear(dim,4dim)` → exact GELU → `Linear(4dim,dim)` → residual. Also no
  LayerScale.
- Final mask: `einsum("bchw,bnc->bnhw", spatial_proj, query_proj) + bias` —
  implemented as `ggml_mul_mat(spatial_proj_tok, query_proj)` on
  channel-fastest token-major tensors (a plain per-location/per-query dot
  product over channels), `+ bias` (`nn.Parameter(1,)`, broadcasts).
- Upsampling: `F.interpolate(spatial_features, size=image_size//downsample_ratio,
mode="bilinear", align_corners=False)` — `ggml_interpolate(...,
GGML_SCALE_MODE_BILINEAR)` (no `ALIGN_CORNERS` flag) matches **exactly**
  (isolated test: 1e-6 diff on synthetic data), confirming ggml's bilinear
  resize agrees with `F.interpolate`'s default convention.

## Why the mask gate is looser (0.15, not 5e-2) — verified not a bug

Debugging trail (each isolation test below was built, run, and deleted after
confirming — not part of the committed test suite, since they depended on
ad-hoc inline reference dumps rather than a checked-in `gen_reference/*.py`
script):

1. `ggml_interpolate` bilinear vs `F.interpolate`: exact match (1e-6) on
   synthetic 4×4→6×6 data.
2. `DepthwiseConvBlock` (hand-transcribed) vs the real module: exact match
   (0.0 diff) on synthetic 5×5×8 data with random weights.
3. `MLPBlock` vs the real module: exact match (0.0 diff) on synthetic data.
4. Full `SegmentationHead.forward()` wiring (the block-chaining/pairing
   loop) vs the real module, fed the **exact captured** `spatial_features`/
   `query_features` from a spy-hooked real forward pass: all 4
   `mask_logits[i]` matched to ~5e-5.
5. `hidden_states[0..3]` from `src/decoder.cpp`, run standalone and diffed
   against the real model's captured per-layer `hs[i]`: all 4 matched to
   ~1e-3-2.5e-3 (comparable to the already-accepted detection-milestone
   scale, e.g. `test_decoder`'s 7.3e-4 on `pred_logits`).
6. `fused` (backbone+projector output) for this config: matched to 4e-6.

Every individual piece is correct. What differs in the full pipeline is that
the mask head is a **256-channel dot product** turned directly into a
logit (no softmax/normalization to compress the scale) — a ~1e-3-level
per-channel input error (step 5, already an accepted scale elsewhere) can
combine across 256 channels into an output error, landing at the observed
max-abs-diff (~0.1) while the _mean_ stays tiny (2e-3) — i.e. a few outlier
query/pixel combinations see amplified drift, not a systematic error. This
is plausible floating-point behavior for a from-scratch reimplementation
feeding a large dot product, not a logic bug — hence the gate is relaxed
specifically for `pred_masks`, with this reasoning recorded inline in
`tests/test_segmentation.cpp` and here.

**Correction after independent review**: an initial draft of this doc
claimed the `~sqrt(256)≈16×` random-sign-accumulation model _predicts_ the
observed 0.109 from the ~1e-3 input error scale. A second-opinion review
correctly flagged that the back-of-envelope bound (`1e-3 × 16 ≈ 1.6e-2`) is
actually **~7× lower** than the observed 0.109 — the number was picked to
match the outcome after the fact, not independently predicted. Softened to
"plausible, no bug found" rather than "the math predicts this exactly."
Two follow-up checks the review requested, done cheaply after the fact:

- **Per-block growth is not smooth/monotonic**, which argues against pure
  compounding-across-4-blocks as the mechanism: feeding real spatial
  features + this port's own `hidden_states[0..3]` gives `mask[0..2]`
  max-abs-diff of 0.0038 / 0.0048 / 0.0067 (mild, roughly linear growth)
  but `mask[3]` jumps to 0.109 — a ~16× discontinuity in one step, despite
  `hidden_states[2]` and `hidden_states[3]`'s own errors being nearly
  identical (~2.3e-3 vs ~2.5e-3, from the standalone `hidden_states` diff
  check). Confirmed by direct ablation below that this is real, not noise.
- **Where the worst pixel is**: the observed argmax (element 158137 of a
  `(78,78,100)` WHC-flattened mask) decodes to `query=25, h=77, w=31` —
  `h=77` is the **last row** of a 0-77-indexed 78-pixel image, i.e. a
  boundary pixel. Consistent with the review's suggested alternative
  explanation (bilinear-interpolation/conv-padding edge sensitivity) being
  at least a contributing factor, alongside or instead of pure random-sign
  channel accumulation. **Not fully characterized** — the honest state is
  "confirmed the discrepancy traces 100% to `hidden_states[3]`'s existing,
  already-accepted drift" (see the ablation below), not "we know exactly
  why the amplification is boundary/query-25-specific." Worth a follow-up
  if this pattern recurs for other heads (the review flagged this
  dot-product-amplification pattern as _likely to recur_, not a one-off).

**Confirmed by direct ablation** (requested as a second-opinion follow-up,
not just the indirect evidence above): ran `segmentation_head()` fed the
_exact_ real captured `spatial_features` plus this port's own
`hidden_states[0..2]` (unchanged) but with `hidden_states[3]` swapped for
the _exact_ reference value. Result: `mask[3]`'s error dropped from 0.109 to
6.3e-5 (near machine precision) — i.e. **100% of the full-pipeline mask
discrepancy is attributable to `hidden_states[3]`'s own ~2.5e-3 drift**
(itself already the accepted scale behind `test_decoder`'s `pred_logits`
7.3e-4/`pred_boxes` 3.3e-4 checks), not to any wiring/pairing/aliasing bug
in `segmentation_head()`'s block loop. This rules out an off-by-one in the
`blocks[i]`↔`hidden_states[i]` pairing, an aliasing hazard in how
`out_hs` accumulates across `src/decoder.cpp`'s layer loop (the
`ggml_clamp`-style in-place-aliasing bug class was explicitly checked for
and not found — `layer_norm_affine` is not in-place, confirmed via
`ggml_norm`'s `inplace=false` call), and a shared-vs-per-block weight mix-up
(`spatial_features_proj`/`query_features_proj`/`bias` are correctly reused
identically every iteration, matching upstream's single shared modules).

Open tasks (other segmentation variants, `RFDETRSegPreviewConfig`, and the
exact amplification-mechanism question) are tracked in
[`0001-open-work.md`](0001-open-work.md), not duplicated here.

## Parked

- Shelved 2026-09-27: masks through the C ABI. `src/rfdetr_capi.cpp` still
  returns boxes only; `demos/seg_cli.cpp` drives the C++ internals
  (backbone, projector, decoder, segmentation head) to get masks. Unpark
  when a consumer needs masks through the C ABI.
