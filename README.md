# interactor-rf-detr-ggml

A C++ and ggml port of the RF-DETR real-time detection transformer, for object detection, instance segmentation and keypoints.

## What it is for

The port runs the converted model without PyTorch at runtime. Each stage, from backbone to heads, is checked against a PyTorch reference by a max-abs-diff test, and Lean proofs under `formal/` cover the backward-pass identities that training relies on. The decisions behind the port are in `docs/decisions/`.

## Build and run

```sh
cmake -B build -G Ninja
cmake --build build
```

The Python tooling, including weight conversion to GGUF and reference generation, runs in the pixi environments that `pixi.toml` declares.

## Licence

The repository does not state a licence for its own code. The vendored ggml under `third_party/` carries its own.
