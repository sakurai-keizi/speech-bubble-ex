# speech_bubble_ex

漫画ページ画像から吹き出しを検出し、吹き出し形状に沿った背景透過PNGとして切り出すツール。

## 仕組み

```
フォルダ内の画像
  ↓
YOLOv11n-seg (huyvux3005/manga109-segmentation-bubble)
  → 吹き出しの検出 + セグメンテーションマスク
  ↓
マスクを元画像サイズに復元
  ↓
RGBA変換（マスク外を透明化）→ マスク範囲でトリミング
  ↓
背景透過PNG として保存
```

モデルは初回実行時に HuggingFace から自動ダウンロードされます（約20MB）。

## 要件

- Python 3.10 以上
- [uv](https://docs.astral.sh/uv/)

依存パッケージは `uv run` 実行時に自動インストールされます。

## 使い方

```bash
# 基本（出力先は自動で <フォルダ名>_bubbles/ になる）
uv run extract_bubbles.py ./chapter01/

# 出力先を指定
uv run extract_bubbles.py ./chapter01/ -o ./output/

# CPU を強制使用
uv run extract_bubbles.py ./chapter01/ --device cpu

# 検出感度を上げる（デフォルト: 0.3）
uv run extract_bubbles.py ./chapter01/ --conf 0.15
```

## 出力ファイル名

```
{フォルダ名}_{ファイル名}_{連番3桁}.png
```

例: `chapter01/page001.webp` に吹き出しが3つある場合

```
chapter01_bubbles/
  chapter01_page001_001.png
  chapter01_page001_002.png
  chapter01_page001_003.png
```

## オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `folder` | (必須) | 入力フォルダ |
| `-o`, `--output` | `<フォルダ名>_bubbles/` | 出力ディレクトリ |
| `--device` | `auto` | 使用デバイス: `auto` / `cuda` / `cpu` |
| `--conf` | `0.3` | 検出信頼度閾値（低くすると検出数増、誤検出も増える） |

## 対応画像形式

`.png` `.jpg` `.jpeg` `.webp` `.bmp` `.tiff`

## モデル

[huyvux3005/manga109-segmentation-bubble](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)

- アーキテクチャ: YOLOv11n-seg
- 学習データ: Manga109 + MangaSegmentation
- mask mAP@50: 99.13%
