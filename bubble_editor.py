#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "PyQt6>=6.4",
#   "Pillow>=10.0",
# ]
# ///
"""
吹き出しエディタ GUI

Usage:
  uv run bubble_editor.py

操作:
  - ベース画像をウインドウにドロップして開く
  - 吹き出しPNGをドロップして追加（ドロップ位置に配置）
  - ドラッグ: 吹き出しを移動
  - 右下コーナーハンドルをドラッグ: 拡大縮小（アスペクト比維持）
  - スクロールホイール: 選択中の吹き出しを拡大縮小
  - Delete / BackSpace: 選択中の吹き出しを削除
  - 「保存」ボタン: 元解像度でコンポジットしてPNG保存
"""

import math
import sys
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QImage, QPainter, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QLabel, QMainWindow, QMessageBox, QPushButton, QToolBar,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
SCROLL_FACTOR = 1.1
HANDLE_SCREEN_PX = 7    # ハンドルの画面上のサイズ（px）
LINE_SCREEN_PX = 1.5    # 選択枠の線幅（px）


# ------------------------------------------------------------------ ユーティリティ

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, rgba.width * 4,
                  QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# ------------------------------------------------------------------ BubbleItem

class BubbleItem(QGraphicsItem):
    """
    吹き出し1枚。シーン座標系 = ベース画像の原寸ピクセル座標。
    pos() が左上、scale で元サイズに対する倍率を管理する。
    """

    def __init__(self, pil_image: Image.Image) -> None:
        super().__init__()
        self.pil_image = pil_image
        self._pixmap = pil_to_qpixmap(pil_image)
        self._scale = 1.0

        self._dragging_handle = False
        self._drag_start_dist = 1.0
        self._drag_start_scale = 1.0

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    # --- QGraphicsItem 必須メソッド ---

    def boundingRect(self) -> QRectF:
        w = self._pixmap.width() * self._scale
        h = self._pixmap.height() * self._scale
        return QRectF(0, 0, w, h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        r = self.boundingRect()
        src = QRectF(self._pixmap.rect())
        painter.drawPixmap(r, self._pixmap, src)

        if self.isSelected():
            vs = self._view_scale()
            lw = LINE_SCREEN_PX / max(vs, 0.001)
            hs = HANDLE_SCREEN_PX / max(vs, 0.001)

            # 選択枠
            pen = QPen(QColor("#4da6ff"), lw, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r)

            # 右下コーナーハンドル（バウンディングボックス内に収める）
            hr = self._handle_rect(hs)
            painter.setPen(QPen(QColor("#4da6ff"), lw))
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.drawRect(hr)

    # --- ヘルパー ---

    def _view_scale(self) -> float:
        scene = self.scene()
        if scene is None:
            return 1.0
        views = scene.views()
        return views[0].transform().m11() if views else 1.0

    def _handle_rect(self, hs: float | None = None) -> QRectF:
        if hs is None:
            hs = HANDLE_SCREEN_PX / max(self._view_scale(), 0.001)
        r = self.boundingRect()
        return QRectF(r.right() - hs * 2, r.bottom() - hs * 2, hs * 2, hs * 2)

    # --- マウスイベント ---

    def mousePressEvent(self, event) -> None:
        if (event.button() == Qt.MouseButton.LeftButton
                and self.isSelected()
                and self._handle_rect().contains(event.pos())):
            self._dragging_handle = True
            # ドラッグ開始時のカーソル位置からの距離を基準にする（ジャンプ防止）
            self._drag_start_dist = max(
                math.hypot(event.pos().x(), event.pos().y()), 0.1
            )
            self._drag_start_scale = self._scale
            event.accept()
            return
        self._dragging_handle = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_handle:
            cur_dist = max(math.hypot(event.pos().x(), event.pos().y()), 0.1)
            new_scale = self._drag_start_scale * cur_dist / self._drag_start_dist
            self.prepareGeometryChange()
            self._scale = max(0.02, new_scale)
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging_handle = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.delta()
        factor = SCROLL_FACTOR if delta > 0 else 1.0 / SCROLL_FACTOR
        self.prepareGeometryChange()
        self._scale = max(0.02, self._scale * factor)
        self.update()
        event.accept()


# ------------------------------------------------------------------ EditorView

class EditorView(QGraphicsView):
    """ベース画像と吹き出しを表示するビュー。ファイルのDnD・キー操作に対応。"""

    def __init__(self, scene: QGraphicsScene, status_fn, parent=None) -> None:
        super().__init__(scene, parent)
        self._set_status = status_fn
        self._base_pil: Image.Image | None = None
        self._base_item: QGraphicsPixmapItem | None = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(QColor("#3c3f41")))

    # --- DnD ---

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                img = Image.open(path).convert("RGBA")
            except Exception as exc:
                QMessageBox.critical(self, "エラー", f"画像を開けませんでした:\n{path}\n{exc}")
                continue

            if self._base_pil is None:
                self._set_base(img, path)
            else:
                drop_scene = self.mapToScene(event.position().toPoint())
                self._add_bubble(img, drop_scene)

    # --- ベース画像 ---

    def _set_base(self, img: Image.Image, path: Path) -> None:
        self._base_pil = img
        pixmap = pil_to_qpixmap(img)
        if self._base_item is None:
            self._base_item = QGraphicsPixmapItem(pixmap)
            self._base_item.setZValue(-1)
            self.scene().addItem(self._base_item)
        else:
            self._base_item.setPixmap(pixmap)
        self.scene().setSceneRect(0, 0, img.width, img.height)
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._set_status(
            f"ベース: {path.name}  ({img.width}×{img.height})"
            "  |  吹き出しPNGをドロップして追加"
        )

    # --- 吹き出し追加 ---

    def _add_bubble(self, img: Image.Image, drop_pos: QPointF) -> None:
        short_side = min(self._base_pil.width, self._base_pil.height)
        max_dim = max(img.width, img.height, 1)
        initial_scale = min(1.0, short_side * 0.20 / max_dim)

        item = BubbleItem(img)
        item._scale = initial_scale
        w = img.width * initial_scale
        h = img.height * initial_scale
        item.setPos(drop_pos.x() - w / 2, drop_pos.y() - h / 2)
        self.scene().addItem(item)
        self.scene().clearSelection()
        item.setSelected(True)

    # --- リセット ---

    def reset(self) -> None:
        """すべてをクリアして起動直後の状態に戻す。"""
        # 吹き出しをすべて削除
        for item in list(self.scene().items()):
            if isinstance(item, BubbleItem):
                self.scene().removeItem(item)
        # ベース画像を削除
        if self._base_item is not None:
            self.scene().removeItem(self._base_item)
            self._base_item = None
        self._base_pil = None
        self.scene().setSceneRect(0, 0, 0, 0)
        self._set_status("ベース画像をウインドウにドロップしてください")

    # --- 削除 ---

    def delete_selected(self) -> None:
        for item in list(self.scene().selectedItems()):
            if isinstance(item, BubbleItem):
                self.scene().removeItem(item)

    # --- 保存 ---

    def save(self) -> None:
        if self._base_pil is None:
            QMessageBox.warning(self, "警告", "ベース画像がありません。")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "保存先を選択", "", "PNG (*.png)"
        )
        if not out_path:
            return

        # 吹き出しのみのレイヤー（ベース画像と同サイズ・透過背景）
        w, h = self._base_pil.width, self._base_pil.height
        bubbles_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        for item in self.scene().items():
            if not isinstance(item, BubbleItem):
                continue
            bw = max(1, int(item.pil_image.width * item._scale))
            bh = max(1, int(item.pil_image.height * item._scale))
            resized = item.pil_image.resize((bw, bh), Image.LANCZOS)
            px = int(item.pos().x())
            py = int(item.pos().y())
            bubbles_layer.paste(resized, (px, py), resized)

        # 合成画像（ベース + 吹き出し）
        result = self._base_pil.copy().convert("RGBA")
        result.paste(bubbles_layer, (0, 0), bubbles_layer)

        out = Path(out_path)
        bubbles_path = out.with_stem(out.stem + "_bubbles_only")

        result.save(out_path, "PNG")
        bubbles_layer.save(bubbles_path, "PNG")

        self._set_status(f"保存しました: {out.name}  /  {bubbles_path.name}")
        QMessageBox.information(
            self, "保存完了",
            f"合成画像:\n{out_path}\n\n吹き出しのみ:\n{bubbles_path}"
        )

    # --- イベントハンドラ ---

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._base_pil is not None:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ------------------------------------------------------------------ MainWindow

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("吹き出しエディタ")
        self.resize(1100, 800)

        scene = QGraphicsScene(self)
        self._status_label = QLabel("ベース画像をウインドウにドロップしてください")
        self._status_label.setStyleSheet("color: #aaaaaa; padding: 0 8px;")

        self._view = EditorView(scene, self._status_label.setText, self)
        self.setCentralWidget(self._view)

        toolbar = QToolBar("ツール", self)
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            "QToolBar { background: #3c3f41; border: none; padding: 4px; spacing: 4px; }"
        )
        self.addToolBar(toolbar)

        btn_style = (
            "QPushButton { background: #4c5052; color: white; border: none;"
            " padding: 4px 14px; border-radius: 3px; }"
            "QPushButton:hover { background: #5c6062; }"
        )
        btn_save = QPushButton("保存")
        btn_save.setStyleSheet(btn_style)
        btn_save.clicked.connect(self._view.save)
        toolbar.addWidget(btn_save)

        btn_del = QPushButton("選択を削除  [Del]")
        btn_del.setStyleSheet(btn_style)
        btn_del.clicked.connect(self._view.delete_selected)
        toolbar.addWidget(btn_del)

        btn_reset = QPushButton("やり直し")
        btn_reset.setStyleSheet(btn_style)
        btn_reset.clicked.connect(self._confirm_reset)
        toolbar.addWidget(btn_reset)

        toolbar.addWidget(self._status_label)

    def _confirm_reset(self) -> None:
        reply = QMessageBox.question(
            self, "やり直し確認",
            "すべてをクリアして最初からやり直しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._view.reset()


# ------------------------------------------------------------------ エントリーポイント

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
