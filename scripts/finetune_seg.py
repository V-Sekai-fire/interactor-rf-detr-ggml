#!/usr/bin/env python3
"""Fine-tune RFDETRSegNano from the upstream COCO checkpoint, keeping the 91-slot COCO head.

Plan step 3 (avatars as persons). Merges one or more COCO-format datasets into a single
COCO-2017-layout tree, trains with ``rfdetr``'s own ``RFDETRSegNano.train()``, and writes a
slim ``{"model": state_dict, "args": ...}`` .pth that the four ``convert_*_to_gguf.py``
scripts read exactly as they read the upstream checkpoint.

WHY THE MERGED TREE IS COCO-2017 LAYOUT, NOT ROBOFLOW LAYOUT. ``dataset_file="roboflow"``
(``train/_annotations.coco.json``) builds ``CocoDetection(remap_category_ids=True)``, which
renumbers the categories to contiguous labels 0..N-1 -- "person" (COCO id 1) would train
the head's slot 0. The upstream checkpoint and the ggml graph index the head BY COCO ID
(``num_classes=90`` -> ``class_embed`` of 91, slot 1 = person). ``dataset_file="coco"``
(``train2017/``, ``val2017/``, ``annotations/instances_{train,val}2017.json``) does not
remap, so slot k keeps meaning COCO id k. ``num_classes=90`` is also passed explicitly, so
``load_pretrain_weights`` and ``_align_num_classes_from_dataset`` both keep the head at 91
instead of re-heading it to the dataset's class count.

INPUTS (``--dataset-dirs``), each one of:
  * Roboflow COCO layout: ``train/``, ``valid/``, ``test/`` each with
    ``_annotations.coco.json`` (what ``coco_person_subset.py`` writes). train -> train,
    valid -> val; ``test`` is never merged, it stays held out.
  * A flat dir with ``annotations.json`` (or ``_annotations.coco.json``) and images at the
    ``file_name`` paths it names (the step-1 render output). All of it goes to train unless
    ``--flat-split val`` says otherwise.
Categories are matched BY NAME onto COCO's 80 ids; a name COCO lacks is an error, not a
silent new class. Image and annotation ids are renumbered from 1 across all inputs, and
image files are renamed ``d<k>_<original path with / -> _>`` so two inputs can never
collide. Annotations without a ``segmentation`` are dropped (the mask loss needs one).

RUNNING ON THE WINDOWS RTX 4090. pixi.exe on a ``\\\\wsl.localhost`` path is slow and has no
symlinks, so the Windows side is a plain copy (the ``train`` env itself lives in pixi's
detached env cache, not in the copy)::

    # from WSL, in rf-detr-ggml/
    W=/mnt/c/Users/ernest.lee/AppData/Local/rfdetr-train
    mkdir -p $W/scripts $W/models $W/data
    cp pixi.toml pixi.lock $W/ && cp scripts/*.py $W/scripts/ && cp models/rf-detr-seg-nano.pt $W/models/
    cp -r data/coco-person $W/data/            # and any render / pseudo-label datasets
    cd $W && /mnt/c/Users/ernest.lee/scoop/shims/pixi.exe run -e train python scripts/finetune_seg.py \\
        --dataset-dirs data/coco-person data/vrm-renders --epochs 20 \\
        --out models/rf-detr-seg-nano-avatar.pth

    # dry run: 1 epoch, 50 train images
    ... finetune_seg.py --dataset-dirs data/coco-person --epochs 1 --max-train 50 --max-val 10 \\
        --out models/rf-detr-seg-nano-dryrun.pth

Then convert (``reference`` env, CPU, from WSL -- same commands as for the upstream .pt)::

    pixi run -e reference python scripts/convert_dinov2_to_gguf.py    X.pth X-backbone.gguf seg-nano
    pixi run -e reference python scripts/convert_projector_to_gguf.py X.pth X-projector.gguf
    pixi run -e reference python scripts/convert_decoder_to_gguf.py   X.pth X-decoder.gguf 4
    pixi run -e reference python scripts/convert_segmentation_to_gguf.py X.pth X-segmentation.gguf
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

SPLIT_DIRS = {"train": "train2017", "val": "val2017"}
ROBOFLOW_SPLITS = {"train": "train", "valid": "val"}
FLAT_ANN_NAMES = ("annotations.json", "_annotations.coco.json")
HEAD_SLOTS = 91  # num_classes=90 + 1; slot index == COCO category id


def coco_categories() -> list[dict]:
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return [{"id": cid, "name": name, "supercategory": "none"} for cid, name in sorted(COCO_CLASSES.items())]


def discover(dataset_dir: Path, flat_split: str) -> list[tuple[str, Path, Path]]:
    """Return (merged_split, annotation_json, image_root) triples for one input dir."""
    found = []
    for src, dst in ROBOFLOW_SPLITS.items():
        ann = dataset_dir / src / "_annotations.coco.json"
        if ann.exists():
            found.append((dst, ann, ann.parent))
    if found:
        return found
    for name in FLAT_ANN_NAMES:
        ann = dataset_dir / name
        if ann.exists():
            return [(flat_split, ann, dataset_dir)]
    raise FileNotFoundError(f"{dataset_dir}: neither train/_annotations.coco.json nor {' / '.join(FLAT_ANN_NAMES)}")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge(dataset_dirs: list[Path], work: Path, flat_split: str, caps: dict[str, int], seed: int) -> dict[str, dict]:
    cats = coco_categories()
    name_to_id = {c["name"]: c["id"] for c in cats}
    out = {s: {"images": [], "annotations": [], "categories": cats} for s in SPLIT_DIRS}
    next_img, next_ann = 1, 1
    dropped_no_mask = 0

    # Gather per split first so --max-train/--max-val cap the MERGED pool, seeded.
    pool: dict[str, list[tuple[int, Path, dict, list[dict], dict[int, int]]]] = {s: [] for s in SPLIT_DIRS}
    for k, d in enumerate(dataset_dirs):
        for split, ann_path, img_root in discover(d, flat_split):
            with open(ann_path, encoding="utf-8") as f:
                doc = json.load(f)
            cat_map = {}
            for c in doc["categories"]:
                if c["name"] not in name_to_id:
                    used = any(a["category_id"] == c["id"] for a in doc["annotations"])
                    if used:
                        raise ValueError(f"{ann_path}: category {c['name']!r} is not a COCO class")
                    continue
                cat_map[c["id"]] = name_to_id[c["name"]]
            anns_by_img: dict[int, list[dict]] = {}
            for a in doc["annotations"]:
                anns_by_img.setdefault(a["image_id"], []).append(a)
            for im in doc["images"]:
                pool[split].append((k, img_root, im, anns_by_img.get(im["id"], []), cat_map))

    rng = random.Random(seed)
    for split, items in pool.items():
        if caps.get(split, 0) > 0 and len(items) > caps[split]:
            rng.shuffle(items)
            items = items[:caps[split]]
        img_dir = work / SPLIT_DIRS[split]
        img_dir.mkdir(parents=True, exist_ok=True)
        for k, img_root, im, anns, cat_map in items:
            src = img_root / im["file_name"]
            new_name = f"d{k}_" + im["file_name"].replace("/", "_").replace("\\", "_")
            link_or_copy(src, img_dir / new_name)
            new_im = {"id": next_img, "file_name": new_name, "width": im["width"], "height": im["height"]}
            out[split]["images"].append(new_im)
            for a in anns:
                if not a.get("segmentation"):
                    dropped_no_mask += 1
                    continue
                out[split]["annotations"].append({
                    "id": next_ann,
                    "image_id": next_img,
                    "category_id": cat_map[a["category_id"]],
                    "bbox": a["bbox"],
                    "area": a.get("area", a["bbox"][2] * a["bbox"][3]),
                    "iscrowd": a.get("iscrowd", 0),
                    "segmentation": a["segmentation"],
                })
                next_ann += 1
            next_img += 1

    ann_dir = work / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split, doc in out.items():
        with open(ann_dir / f"instances_{SPLIT_DIRS[split]}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f)
        n_person = sum(1 for a in doc["annotations"] if a["category_id"] == 1)
        print(f"merged {split}: {len(doc['images'])} images, {len(doc['annotations'])} annotations "
              f"({n_person} person)", flush=True)
    if dropped_no_mask:
        print(f"dropped {dropped_no_mask} annotations without a segmentation", flush=True)
    if not out["train"]["images"] or not out["val"]["images"]:
        raise ValueError("merged dataset needs at least one train and one val image")
    return out


class StepTimer:
    """Wraps RFDETRModelModule.training_step to time micro-batches (CUDA-synchronised)."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self._last: float | None = None

    def install(self) -> None:
        import torch
        from rfdetr.training.module_model import RFDETRModelModule

        orig = RFDETRModelModule.training_step
        timer = self

        def timed(module, batch, batch_idx):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            if timer._last is not None and batch_idx > 0:
                timer.times.append(now - timer._last)
            timer._last = now
            return orig(module, batch, batch_idx)

        RFDETRModelModule.training_step = timed


def pick_checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "last_ema.pth"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no rfdetr checkpoint in {run_dir}")


def export(ckpt_path: Path, pretrain: Path, out: Path) -> None:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    ref = torch.load(pretrain, map_location="cpu", weights_only=False)
    ref_sd = ref["model"] if "model" in ref else ref
    if sd["class_embed.bias"].shape[0] != HEAD_SLOTS:
        raise ValueError(f"class_embed has {sd['class_embed.bias'].shape[0]} slots, expected {HEAD_SLOTS}")
    missing = sorted(set(ref_sd) - set(sd))
    shape_diff = sorted(k for k in set(ref_sd) & set(sd) if tuple(ref_sd[k].shape) != tuple(sd[k].shape))
    if missing or shape_diff:
        raise ValueError(f"state dict drifted from the pretrain layout: missing={missing[:5]} shapes={shape_diff[:5]}")
    extra = sorted(set(sd) - set(ref_sd))
    if extra:
        print(f"note: {len(extra)} keys not in the pretrain checkpoint (ignored by converters): {extra[:5]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "args": ckpt.get("args"), "epoch": ckpt.get("epoch"),
                "source_checkpoint": ckpt_path.name, "pretrain": pretrain.name}, out)
    print(f"wrote {out} ({len(sd)} tensors, from {ckpt_path.name}, epoch {ckpt.get('epoch')})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True, help="slim .pth for the convert_*_to_gguf.py scripts")
    ap.add_argument("--pretrain", type=Path, default=Path("models/rf-detr-seg-nano.pt"),
                    help="upstream rf-detr-seg-n-ft.pth (md5 9995497791d0ff1664a1d9ddee9cfd20)")
    ap.add_argument("--work-dir", type=Path, default=None, help="merged dataset + rfdetr output (default: data/runs/<out stem>)")
    ap.add_argument("--flat-split", choices=["train", "val"], default="train")
    ap.add_argument("--max-train", type=int, default=0, help="cap merged train images (0 = all)")
    ap.add_argument("--max-val", type=int, default=0, help="cap merged val images (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu", default="0",
                    help="CUDA_VISIBLE_DEVICES value (0 = the 4090 on the Windows desk). rfdetr 1.9.4's "
                         "build_trainer crashes on device='cuda:N' (devices=[N] hits .strip()), so the GPU is "
                         "chosen by visibility and rfdetr sees exactly one device.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # rfdetr's rich metrics table crashes a cp1252 Windows console (UnicodeEncodeError at the
    # end of validation), so force UTF-8 on the streams before anything prints.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    import torch
    from rfdetr import RFDETRSegNano

    work = args.work_dir or Path("data/runs") / args.out.stem
    data_dir, run_dir = work / "dataset", work / "rfdetr"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    merge(args.dataset_dirs, data_dir, args.flat_split, {"train": args.max_train, "val": args.max_val}, args.seed)

    # num_classes=90 EXPLICIT: marks it user-set, so neither the checkpoint loader nor the
    # dataset alignment re-heads the model; 90 + background = the checkpoint's 91 slots.
    model = RFDETRSegNano(pretrain_weights=str(args.pretrain), num_classes=90)

    timer = StepTimer()
    timer.install()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.train(
        dataset_dir=str(data_dir),
        dataset_file="coco",
        output_dir=str(run_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.num_workers,
        tensorboard=False,
        progress_bar="tqdm",
        seed=args.seed,
    )
    wall = time.perf_counter() - t0

    if timer.times:
        ts = sorted(timer.times)
        med = ts[len(ts) // 2]
        print(f"timing: {len(ts) + 1} micro-batches of {args.batch_size}, median {med * 1000:.0f} ms/it, "
              f"mean {sum(ts) / len(ts) * 1000:.0f} ms/it, total train() wall {wall:.1f} s", flush=True)
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print(f"vram: peak allocated {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB, "
              f"peak reserved {torch.cuda.max_memory_reserved(dev) / 2**30:.2f} GiB "
              f"on {torch.cuda.get_device_name(dev)}", flush=True)

    export(pick_checkpoint(run_dir), args.pretrain, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
