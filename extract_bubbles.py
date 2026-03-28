#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "ultralytics>=8.0",
#   "huggingface-hub>=0.20",
#   "Pillow>=10.0",
#   "numpy>=1.24",
#   "torch>=2.0",
#   "torchvision>=0.15",
# ]
# ///
"""
漫画ページから吹き出しを切り出して背景透過PNGとして保存するスクリプト。

Pipeline:
  YOLOv11n-seg (huyvux3005/manga109-segmentation-bubble) で吹き出しの
  セグメンテーションマスクを取得し、背景透過PNGとして切り出す。

Usage:
  uv run extract_bubbles.py <folder> [-o <output_dir>] [--conf 0.3] [--device auto|cuda|cpu]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter
from ultralytics import YOLO

YOLO_HF_REPO = "huyvux3005/manga109-segmentation-bubble"
OUTLINE_RATIO = 0.015   # 縁取り厚さ = bbox縦横平均の1.5%
YOLO_HF_FILE = "best.pt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> YOLO:
    print(f"モデルをダウンロード中: {YOLO_HF_REPO}/{YOLO_HF_FILE}")
    model_path = hf_hub_download(repo_id=YOLO_HF_REPO, filename=YOLO_HF_FILE)
    return YOLO(model_path)


def process_image(
    image_path: Path,
    folder_name: str,
    output_dir: Path,
    model: YOLO,
    device: str,
    conf: float,
) -> int:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    print(f"\n処理中: {image_path.name}  ({image.width}x{image.height})")

    results = model(image_np, verbose=False, conf=conf, device=device)[0]

    if results.masks is None or len(results.masks) == 0:
        print("  吹き出しが検出されませんでした。")
        return 0

    masks_data = results.masks.data.cpu().numpy()   # (N, H, W) float32 0〜1
    boxes = results.boxes.xyxy.cpu().numpy()         # (N, 4) xyxy
    print(f"  {len(masks_data)} 個の吹き出しを検出しました。")

    h_orig, w_orig = image_np.shape[:2]
    saved = 0

    for i, (mask_raw, box) in enumerate(zip(masks_data, boxes)):
        # マスクを元画像サイズにリサイズ
        mask_img = Image.fromarray((mask_raw * 255).astype(np.uint8), mode="L")
        if mask_img.size != (w_orig, h_orig):
            mask_img = mask_img.resize((w_orig, h_orig), Image.NEAREST)
        mask = np.array(mask_img) > 127  # (H, W) bool

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        # マスクの bounding box
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

        # 縁取り: Gaussian ブラーでエッジからの距離を連続値化し滑らかに合成
        #   b = GaussianBlur(mask)     → エッジ付近で 0〜1 の滑らかな勾配になる
        #   border_alpha = 4*b*(1-b)   → エッジ上で 1.0、内外に向かって 0 の山型
        thickness = max(1, int(((x2 - x1) + (y2 - y1)) / 2 * OUTLINE_RATIO))
        blur_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        blurred = np.array(
            blur_img.filter(ImageFilter.GaussianBlur(radius=float(thickness)))
        ).astype(np.float32) / 255.0
        border_alpha = 4.0 * blurred * (1.0 - blurred)  # (H, W) range [0, 1]

        # RGBA を float で組み立てて黒縁をアルファブレンド
        rgba = np.zeros((h_orig, w_orig, 4), dtype=np.float32)
        rgba[..., :3] = image_np.astype(np.float32)
        rgba[..., 3]  = mask.astype(np.float32) * 255.0
        ba = border_alpha[..., np.newaxis]               # (H, W, 1)
        rgba[..., :3] = rgba[..., :3] * (1.0 - ba)      # エッジに近いほど黒に近づける
        rgba[..., 3]  = np.maximum(rgba[..., 3], border_alpha * 255.0)  # 縁を不透明に
        rgba = rgba.astype(np.uint8)
        cropped = rgba[y1:y2, x1:x2]

        out_path = output_dir / f"{folder_name}_{image_path.stem}_{i+1:03d}.png"
        Image.fromarray(cropped, "RGBA").save(out_path)
        print(f"  保存: {out_path.name}  (size: {x2-x1}x{y2-y1})")
        saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="漫画フォルダ内の全画像から吹き出しを背景透過PNGとして切り出す",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", type=Path, help="入力フォルダ")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="出力ディレクトリ（省略時: <フォルダ名>_bubbles/）",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="使用デバイス",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="検出信頼度閾値（0〜1）",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"エラー: フォルダが見つかりません: {args.folder}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        p for p in args.folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"エラー: 対応する画像ファイルが見つかりません: {args.folder}", file=sys.stderr)
        sys.exit(1)

    device = resolve_device(args.device)
    folder_name = args.folder.resolve().name
    output_dir = args.output or args.folder.parent / f"{folder_name}_bubbles"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"デバイス: {device}")
    print(f"入力フォルダ: {args.folder}  ({len(image_files)} 枚)")
    print(f"出力フォルダ: {output_dir}")

    model = load_model()

    total_saved = 0
    for image_path in image_files:
        total_saved += process_image(image_path, folder_name, output_dir, model, device, args.conf)

    print(f"\n全完了: 合計 {total_saved} 枚を {output_dir} に保存しました。")


if __name__ == "__main__":
    main()
