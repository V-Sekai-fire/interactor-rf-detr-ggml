# rf-detr-cpp

A C++/[ggml](https://github.com/ggml-org/ggml) port of
[RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow, Apache-2.0 for
Nano–Large/Seg/Keypoint; PML 1.0 for XL/2XL): a DINOv2-backbone real-time
detection transformer supporting object detection, instance segmentation, and
keypoint detection. Modeled on
[trellis2cpp](https://github.com/weftspun/trellis2cpp) and
[see-through-cpp](https://github.com/weftspun/see-through-cpp): GGUF
conversion first, per-milestone validated ggml graphs, no PyTorch at runtime.

See [docs/architecture.md](docs/architecture.md) for the upstream model
architecture notes this port is working from,
[docs/decisions/](docs/decisions/) for divergence/port-decision records, and
[docs/decisions/0001-open-work.md](docs/decisions/0001-open-work.md) for the
consolidated open-task checklist across all milestones.

[formal/rfdetr_proofs/](formal/rfdetr_proofs/) is a small Lean4/Mathlib
project that formally verifies (not just numerically diff-tests) the
backward-pass primitive tricks Phase 2 training relies on — see
[docs/decisions/0004-formal-verification.md](docs/decisions/0004-formal-verification.md).

## Status

| Milestone                                                                       | Test                        | Result                                                                                     |
| ------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------ |
| DINOv2-windowed-attention backbone (RFDETRNano, 4 taps)                         | `test_backbone`             | ≤2.5e-4 (gate 5e-2)                                                                        |
| Multi-scale projector (C2f fusion, RFDETRNano)                                  | `test_projector`            | 1.0e-5                                                                                     |
| Deformable attention core (isolated, synthetic)                                 | `test_deform_attn`          | 0.0 (exact)                                                                                |
| **Object detection end-to-end (RFDETRNano)**                                    | `test_decoder`              | boxes 3.3e-4, logits 7.3e-4                                                                |
| **Instance segmentation end-to-end (RFDETRSegNano)**                            | `test_segmentation`         | boxes 1.1e-2, logits 2.9e-3, masks 0.109 (gate 0.15, see `docs/decisions/segmentation.md`) |
| **Keypoint detection end-to-end (RFDETRKeypointPreview)**                       | `test_keypoints`            | boxes 3.5e-3, logits 9.3e-4, keypoints 4.2e-3                                              |
| RFDETRBase backbone (patch_size==14, bicubic+antialias pos-embed interpolation) | `test_backbone_base`        | ≤1.6e-4                                                                                    |
| **RFDETRBase object detection end-to-end**                                      | `test_decoder_base`         | boxes 1.1e-3, logits 7.2e-4                                                                |
| **RFDETRSmall object detection end-to-end**                                     | `test_decoder_small`        | boxes 4.5e-4, logits 9.3e-4                                                                |
| **RFDETRMedium object detection end-to-end**                                    | `test_decoder_medium`       | boxes 1.8e-3, logits 2.2e-3                                                                |
| **RFDETRLarge object detection end-to-end**                                     | `test_decoder_large`        | boxes 1.5e-3, logits 1.8e-3                                                                |
| **RFDETRSegSmall instance segmentation end-to-end**                             | `test_segmentation_small`   | boxes 4.5e-4, logits 4.9e-4, masks 4.9e-2 (gate 0.15)                                      |
| **RFDETRSegMedium instance segmentation end-to-end**                            | `test_segmentation_medium`  | boxes 6.2e-3, logits 4.1e-3, masks 8.4e-2 (gate 0.15)                                      |
| **RFDETRSegLarge instance segmentation end-to-end**                             | `test_segmentation_large`   | boxes 2.4e-3, logits 7.1e-3, masks 7.8e-2 (gate 0.15)                                      |
| **RFDETRSegPreview instance segmentation end-to-end**                           | `test_segmentation_preview` | boxes 4.7e-4, logits 1.6e-3, masks 5.6e-2 (gate 0.15)                                      |

**All three inference milestones are done for the Nano-family variants:
object detection, instance segmentation, and keypoint detection.** Now
extending to other model-size variants — see `docs/decisions/0001-open-work.md`.

1. ~~Backbone: DINOv2-with-windowed-attention encoder~~ — validated for
   RFDETRNano, RFDETRSegNano, RFDETRKeypointPreview, and RFDETRBase's
   backbone (including bicubic+antialias position-embedding interpolation,
   see `docs/decisions/0002-position-embed-bicubic.md`); Small/Large/XL/2XL
   still to do, and RFDETRBase's projector+decoder aren't wired up yet.
2. ~~Detection head: Deformable-DETR-style decoder~~ — validated end-to-end
   (backbone → projector → decoder → boxes/logits) for RFDETRNano.
3. ~~Instance segmentation head~~ — validated end-to-end for RFDETRSegNano
   (dot-product mask head over all decoder-layer query features).
4. ~~Keypoint head~~ — validated end-to-end for RFDETRKeypointPreview (dual
   projector, AdaLN-modulated GroupPose keypoint decoder stream).
5. Finetuning/training (C++/ggml training loop) — phase 2, after 1-4 are
   validated for inference

Each milestone is validated against a PyTorch reference
(`gen_reference/*.py` via `pixi run -e reference` CPU torch → `tests/test_*.cpp` max-abs-diff
gate), following the sibling ports' pattern:

```sh
cmake -B build -G Ninja && cmake --build build --target test_backbone
pixi run -e reference python scripts/convert_dinov2_to_gguf.py models/rf-detr-nano.pth models/rf-detr-nano-backbone.gguf
pixi run -e reference python gen_reference/gen_reference_backbone.py models/rf-detr-nano.pth gen_reference/reference_backbone_nano.bin
./build/test_backbone.exe
```

Checkpoints download from `https://storage.googleapis.com/rfdetr/*` (see
`docs/decisions/backbone-windowing.md`); `models/` is gitignored.

## Segmenting images (`seg_cli`)

`demos/seg_cli.cpp` runs RFDETRSegNano instance segmentation on
preprocessed frames (312×312×3 float32, planar, ImageNet-normalised) and
writes `<input>.seg` with boxes, class logits and 78×78 mask logits per
query. It is the flat control for interactor-dress-on's gate 9.

```sh
cmake -B build -G Ninja -DGGML_OPENMP=OFF && cmake --build build --target seg_cli
./build/seg_cli MODEL_DIR IMAGE.f32 [IMAGE.f32 ...]
```

It drives the C++ internals because the C ABI still returns boxes only
(parked, see `docs/decisions/segmentation.md`). The avatar fine-tune
scripts (`scripts/coco_person_subset.py`, `scripts/finetune_seg.py`,
`pixi run -e train`) work but the full run is parked, see
`docs/decisions/0003-training.md`.

## Weights

Converted GGUF weights (F32) for every checkpoint-validated variant are
published on this repo's GitHub Releases:
[v0.1.0-dev](https://github.com/weftspun/rf-detr-cpp/releases/tag/v0.1.0-dev).
Each variant is split into separate `<variant>-{backbone,projector,decoder}
[,-segmentation][,-keypoints][,-cross-projector]` files — load all parts for
a variant into one `Model` via repeated `rfdetr_load()` calls. All files are
under GitHub's 2GB asset limit, so none needed zstd splitting (mirroring
see-through-cpp's release layout otherwise). RFDETRSegXLarge/Seg2XLarge are
NOT published: XL/2XL upstream checkpoints are PML 1.0-licensed rather than
Apache-2.0 (SegXLarge's own validation now passes, see
`docs/decisions/0001-open-work.md` — licensing is the only remaining reason
these aren't published).

Upstream PyTorch checkpoints are plain `.pth` state dicts hosted at
`https://storage.googleapis.com/rfdetr/*.pth` (not HuggingFace) — not
re-hosted here; convert them yourself with `scripts/convert_*.py` if you
need to reproduce or extend the GGUF conversion.
