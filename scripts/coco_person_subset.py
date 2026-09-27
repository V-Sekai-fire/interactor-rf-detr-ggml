#!/usr/bin/env python3
"""Cut a COCO "images with people" subset, written as a Roboflow-style COCO split.

Plan step 3 mixes ~40% real COCO persons into the avatar fine-tune so RFDETRSegNano does
not forget real people. This script produces that share:

  1. Downloads ``annotations_trainval2017.zip`` from the official COCO host (once, into
     ``--download-dir``) and reads ``instances_<--source>.json`` straight out of the zip.
  2. Keeps images carrying at least one non-crowd ``category_id == 1`` (person) instance,
     and whose Flickr licence is in ``--licenses`` (see LICENCES below).
  3. Keeps EVERY annotation on those images, all 80 COCO categories, with the original COCO
     category ids -- the fine-tune keeps the 91-slot COCO head, so a chair on the same image
     is still a chair, not background.
  4. Fetches each kept image from its official ``coco_url`` (images.cocodataset.org), or
     from ``<--source>.zip`` in ``--download-dir`` if that is already on disk.
  5. Writes ``<out>/{train,valid,test}/`` each holding its images and an
     ``_annotations.coco.json`` -- the Roboflow COCO layout ``rfdetr`` documents and
     ``scripts/finetune_seg.py --dataset-dirs`` reads.

LICENCES. COCO's annotations are CC-BY 4.0, but each image keeps its Flickr licence, and
three of the eight are NonCommercial. The plan's constraint is CC0/CC-BY only, so the
default keeps licence ids 4 (CC-BY 2.0), 7 (no known copyright restrictions) and 8 (US
Government work). Pass ``--licenses 1 2 3 4 5 6 7 8`` to take everything.

HOW MANY THAT LEAVES (non-crowd person images, measured 2026-09-27):
  val2017:   2693 with a person, 484 under licences 4/7/8 (479 CC-BY + 5 no-known).
  train2017: 64115 with a person, 10704 under licences 4/7/8.
So val2017 alone cannot reach ~2000 CC-BY images; ``--source train2017`` can, and it also
keeps val2017 untouched for the plan's "COCO val persons AP must not drop > 2" check --
training on val2017 images would contaminate that check. The ``test`` split here is held
out either way.

Usage:
    pixi run -e train python scripts/coco_person_subset.py --out data/coco-person-val2017 --max-images 0
    pixi run -e train python scripts/coco_person_subset.py --source train2017 --out data/coco-person-train2017 \
        --max-images 2000
(stdlib only, so plain ``python3`` works too.)
"""
import argparse
import concurrent.futures
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
PERSON_ID = 1
DEFAULT_LICENSES = [4, 7, 8]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def load_instances(download_dir: Path, source: str) -> dict:
    zpath = download_dir / "annotations_trainval2017.zip"
    download(ANNOTATIONS_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        with z.open(f"annotations/instances_{source}.json") as f:
            return json.load(f)


def select_images(coco: dict, source: str, licenses: set[int], max_images: int, seed: int) -> list[dict]:
    person_imgs = set()
    for a in coco["annotations"]:
        if a["category_id"] == PERSON_ID and not a.get("iscrowd", 0):
            person_imgs.add(a["image_id"])
    imgs = [im for im in coco["images"] if im["id"] in person_imgs and im["license"] in licenses]
    imgs.sort(key=lambda im: im["id"])
    print(f"{source}: {len(coco['images'])} images, {len(person_imgs)} with a non-crowd person, "
          f"{len(imgs)} of those under licences {sorted(licenses)}", flush=True)
    rng = random.Random(seed)
    rng.shuffle(imgs)
    if max_images > 0:
        imgs = imgs[:max_images]
    return imgs


def split_images(imgs: list[dict], fractions: tuple[float, float, float]) -> dict[str, list[dict]]:
    n = len(imgs)
    n_train = round(n * fractions[0])
    n_valid = round(n * fractions[1])
    if n >= 3:
        n_valid = max(n_valid, 1)
        n_train = min(n_train, n - n_valid - 1)
    return {
        "train": imgs[:n_train],
        "valid": imgs[n_train:n_train + n_valid],
        "test": imgs[n_train + n_valid:],
    }


def fetch_images(imgs: list[dict], dest: Path, download_dir: Path, source: str, workers: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zpath = download_dir / f"{source}.zip"
    todo = [im for im in imgs if not (dest / im["file_name"]).exists()]
    if not todo:
        return
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            for im in todo:
                with z.open(f"{source}/{im['file_name']}") as src, open(dest / im["file_name"], "wb") as out:
                    shutil.copyfileobj(src, out)
        return

    def one(im: dict) -> None:
        out = dest / im["file_name"]
        tmp = out.with_suffix(".part")
        with urllib.request.urlopen(im["coco_url"], timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(out)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 200 == 0:
                print(f"  {dest.name}: {i}/{len(todo)} images", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/coco-person"))
    ap.add_argument("--source", choices=["val2017", "train2017"], default="val2017",
                    help="COCO split to cut from; train2017 has ~22x more CC-BY person images")
    ap.add_argument("--download-dir", type=Path, default=Path("data/coco-download"))
    ap.add_argument("--max-images", type=int, default=2000, help="0 = every eligible image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--licenses", type=int, nargs="+", default=DEFAULT_LICENSES,
                    help="COCO licence ids to keep (default: 4 CC-BY, 7 no known restrictions, 8 US Gov)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    coco = load_instances(args.download_dir, args.source)
    imgs = select_images(coco, args.source, set(args.licenses), args.max_images, args.seed)
    if not imgs:
        print("no images selected", file=sys.stderr)
        return 1
    splits = split_images(imgs, tuple(args.split))

    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    licenses = [lic for lic in coco["licenses"] if lic["id"] in set(args.licenses)]

    manifest = {"source": f"COCO {args.source}", "annotations_url": ANNOTATIONS_URL, "seed": args.seed,
                "licenses": sorted(args.licenses), "splits": {}}
    for name, split_imgs in splits.items():
        d = args.out / name
        fetch_images(split_imgs, d, args.download_dir, args.source, args.workers)
        split_anns = [a for im in split_imgs for a in anns_by_img.get(im["id"], [])]
        doc = {
            "info": dict(coco["info"], description=f"COCO {args.source} person subset ({name})"),
            "licenses": licenses,
            "categories": coco["categories"],
            "images": split_imgs,
            "annotations": split_anns,
        }
        with open(d / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump(doc, f)
        n_person = sum(1 for a in split_anns if a["category_id"] == PERSON_ID)
        print(f"{name}: {len(split_imgs)} images, {len(split_anns)} annotations ({n_person} person)", flush=True)
        manifest["splits"][name] = sorted(im["id"] for im in split_imgs)

    with open(args.out / "subset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
