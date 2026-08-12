#!/usr/bin/env python3
"""
Kitting Camera Inspection — Trolley Parts Verification
ADDED CUSTOM DISPLAY NAMES in PART_DISPLAY_NAMES below.

Run:
    pip install PySide6 opencv-python ultralytics
    python inspection_ui.py
"""

import sys, math
import os
import cv2
import numpy as np

from PySide6.QtCore    import Qt, QThread, Signal, QTimer, QPointF, QRectF, QRect
from PySide6.QtGui     import (QImage, QPixmap, QFont, QPainter, QColor, QPen,
                                QBrush, QPolygonF, QPainterPath)
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QPushButton, QFrame, QSizePolicy, QFileDialog, QProgressBar
)

# ─────────────────────────── CONFIG ────────────────────────────────
MODEL_PATH          = "best (3).pt"
CAMERA_SOURCE       = "rtsp://admin:Tml%40812004@192.168.1.64:554/Streaming/Channels/101"
CONF_THRESHOLD      = 0.35
RESCAN_SECONDS      = 10
FULLSCREEN          = True
START_IN_IMAGE_MODE = True
ROI_FRAC            = (0.05, 0.05, 0.95, 0.95)
# ───────────────────────────────────────────────────────────────────

# ── CUSTOM DISPLAY NAMES ────────────────────────────────────────────
PART_DISPLAY_NAMES = {
   "left_bracket1"  : "CAB MOUNTING LH",
    "left_bracket 2"  : "GEAR BOX BRACKET LH",
    "centre1"        : "ENGINE MOUNTING PAD LH",
    "centre2"        : "ENGINE MOUNTING PAD RH",
    "centre_triangle": "FILTER BRACKET",
    "centre_circle1" : "ENGINE MTG RUBBER PAD LH",
    "centre_circle2" : "ENGINE MTG RUBBER PAD RH",
    "right1"         : "GEAR BOX BRACKET RH",
    "right2"         : "CAB MOUNTING RH",
    "bottom"         : "FOOT STEP BUMPER MTG",
}
# ───────────────────────────────────────────────────────────────────

# Ordered list — top-to-bottom in the callout column
ALL_PARTS = [
    "left_bracket1", "left_bracket 2",
    "centre1", "centre2", "centre_triangle", "centre_circle1", "centre_circle2",
    "right1", "right2",
    "bottom",
]

# Group separator indices (insert divider BEFORE this index)
GROUP_BREAKS = {2, 7, 9}   # before centre1, right1, bottom

PART_TO_BLOCK = {
    "left_bracket1"  : "BLOCK 1",
    "left_bracket 2"  : "BLOCK 1",
    "centre1"        : "BLOCK 2",
    "centre2"        : "BLOCK 2",
    "centre_triangle": "BLOCK 2",
    "centre_circle1" : "BLOCK 2",
    "centre_circle2" : "BLOCK 2",
    "right1"         : "BLOCK 3",
    "right2"         : "BLOCK 3",
    "bottom"         : "BLOCK 4",
}

BLOCKS = ["BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"]

C_OK    = QColor(61,  220, 132)
C_BAD   = QColor(255,  92,  92)
C_DIM   = QColor(120, 132, 148)
C_BG    = "#0d1016"
C_PANEL = "#141923"
C_TEXT  = "#e8edf2"

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ══════════════════════════ DETECTION ══════════════════════════════

def evaluate_frame(model, frame_bgr):
    results     = model(frame_bgr, conf=CONF_THRESHOLD * 0.3, verbose=False)[0]
    names       = results.names
    found_parts = {}

    for box in results.boxes:
        cls  = names[int(box.cls[0])]
        conf = float(box.conf[0])
        if cls in ALL_PARTS and conf >= CONF_THRESHOLD:
            if cls not in found_parts or conf > found_parts[cls]:
                found_parts[cls] = conf

    # Each part: seen anywhere in frame = OK, not seen = NOT OK
    part_results = {}
    for part in ALL_PARTS:
        part_results[part] = {
            "ok"  : part in found_parts,
            "conf": found_parts.get(part, 0.0),
        }

    annotated  = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)
    any_not_ok = any(not v["ok"] for v in part_results.values())
    return annotated, part_results, any_not_ok


# ══════════════════════════ WORKER ═════════════════════════════════

class ScanWorker(QThread):
    model_ready = Signal(bool, str)
    scan_result = Signal(np.ndarray, dict, bool)   # removed block_ok
    scan_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.model    = None
        self._request = False
        self._image   = None
        self._running = True

    def run(self):
        if not YOLO_AVAILABLE:
            self.model_ready.emit(False, "ultralytics not installed"); return
        try:
            self.model = YOLO(MODEL_PATH)
            self.model_ready.emit(True, "Model loaded")
        except Exception as e:
            self.model_ready.emit(False, f"Model error: {e}"); return

        while self._running:
            if self._request:
                self._request = False
                try:
                    frame = self._image if self._image is not None else self._grab()
                    self._image = None
                    if frame is None:
                        self.scan_failed.emit("No frame (check RTSP / camera)")
                    else:
                        rgb, pr, bad = evaluate_frame(self.model, frame)
                        self.scan_result.emit(rgb, pr, bad)
                except Exception as e:
                    self.scan_failed.emit(str(e))
            self.msleep(40)

    def _grab(self):
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened(): return None
        ok, frame = cap.read()
        for _ in range(3): ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def request_camera_scan(self):   self._image = None; self._request = True
    def request_image_scan(self, f): self._image = f;    self._request = True
    def stop(self):                  self._running = False; self.wait(3000)


# ══════════════════════════ TATA MOTORS LOGO ═══════════════════════
LOGO_CANDIDATES = ["logo1.png", "logo1.jpg", "logo.jpeg", "logo.webp", "logo.bmp"]

class TataMotorsLogo(QFrame):
    """Top-right logo holder. Auto-fetches a logo image from the workspace
    folder on startup; click to swap with another file."""

    def __init__(self):
        super().__init__()

        self.setFixedSize(800, 100)

        self.img = QLabel(self)
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setGeometry(70, 20, 250, 50)
        self.img.setStyleSheet("border:none;")
        self.img.setText("LOGO")
        self._auto_load()

    def _auto_load(self):
        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()

        for name in LOGO_CANDIDATES:
            path = os.path.join(base, name)
            if os.path.exists(path):
                self.set_logo(path)
                return

        for name in LOGO_CANDIDATES:
            if os.path.exists(name):
                self.set_logo(name)
                return

    def set_logo(self, path):
        pix = QPixmap(path)
        if pix.isNull():
            return
        scaled_pix = pix.scaled(
            self.img.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.img.setPixmap(scaled_pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.img.pixmap() and not self.img.pixmap().isNull():
            self.img.setPixmap(
                self.img.pixmap().scaled(
                    self.img.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Logo",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            self.set_logo(path)


# ══════════════════════════ 3D TROLLEY VIEW ════════════════════════

class TrolleyView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 440)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._part_results = {}

    def set_state(self, part_results):
        self._part_results = part_results
        self.update()

    def reset_state(self):
        self._part_results = {}
        self.update()

    # ── iso helpers ───────────────────────────────────────────────
    def _iso(self, x, y, z, cx, cy, s):
        return QPointF(cx+(x-y)*s*0.87, cy-z*s+(x+y)*s*0.5)

    def _poly(self, p, pts, brush, pen=None):
        p.setPen(pen if pen else Qt.NoPen)
        p.setBrush(QBrush(brush))
        p.drawPolygon(QPolygonF(pts))

    def _quad(self, p, x0, y0, x1, y1, z, cx, cy, s, brush, pen=None):
        self._poly(p,
            [self._iso(x0,y0,z,cx,cy,s), self._iso(x1,y0,z,cx,cy,s),
             self._iso(x1,y1,z,cx,cy,s), self._iso(x0,y1,z,cx,cy,s)],
            brush, pen)

    def _box_face(self, p, x0, y0, x1, y1, zb, zt, cx, cy, s,
                  col_top, col_left, col_right):
        self._poly(p,
            [self._iso(x0,y0,zt,cx,cy,s), self._iso(x1,y0,zt,cx,cy,s),
             self._iso(x1,y1,zt,cx,cy,s), self._iso(x0,y1,zt,cx,cy,s)],
            col_top, QPen(col_top.lighter(130), 1))
        self._poly(p,
            [self._iso(x0,y0,zb,cx,cy,s), self._iso(x0,y0,zt,cx,cy,s),
             self._iso(x0,y1,zt,cx,cy,s), self._iso(x0,y1,zb,cx,cy,s)],
            col_left, QPen(col_left.darker(150), 1))
        self._poly(p,
            [self._iso(x0,y1,zb,cx,cy,s), self._iso(x0,y1,zt,cx,cy,s),
             self._iso(x1,y1,zt,cx,cy,s), self._iso(x1,y1,zb,cx,cy,s)],
            col_right, QPen(col_right.darker(150), 1))

    def _post(self, p, x, y, z_top, z_bot, cx, cy, s, w=0.2, col=QColor("#7d93ab")):
        pts = [self._iso(x-w,y-w,z_top,cx,cy,s), self._iso(x+w,y+w,z_top,cx,cy,s),
               self._iso(x+w,y+w,z_bot,cx,cy,s), self._iso(x-w,y-w,z_bot,cx,cy,s)]
        p.setPen(QPen(col.darker(150), 1)); p.setBrush(QBrush(col))
        p.drawPolygon(QPolygonF(pts))

    # ── single callout box + arrow ────────────────────────────────
    def _draw_callout(self, p, tip: QPointF, box_x: float, box_y: float,
                       label: str, ok_state):
        BOX_W, BOX_H = 195, 22

        if ok_state is True:
            col  = QColor(61,  220, 132)
            fill = QColor(15,   50,  30, 225)
            tc   = QColor(61,  220, 132)
            pfx  = "✓  "
        elif ok_state is False:
            col  = QColor(255,  80,  80)
            fill = QColor(60,   10,  10, 225)
            tc   = QColor(255, 110, 110)
            pfx  = "✕  "
        else:
            col  = QColor(0,   195, 240)
            fill = QColor(0,    35,  55, 215)
            tc   = QColor(130, 220, 255)
            pfx  = "·  "

        box_lx = box_x
        box_cy = box_y
        mid_x  = box_lx - 18

        elbow  = QPointF(mid_x, tip.y())
        corner = QPointF(mid_x, box_cy)
        entry  = QPointF(box_lx, box_cy)

        p.setPen(QPen(col, 1.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(tip, elbow)
        p.drawLine(elbow, corner)
        p.drawLine(corner, entry)

        p.setPen(Qt.NoPen); p.setBrush(QBrush(col))
        ah, aw = 8, 4
        p.drawPolygon(QPolygonF([
            QPointF(entry.x() + ah, entry.y()),
            QPointF(entry.x(),      entry.y() - aw),
            QPointF(entry.x(),      entry.y() + aw),
        ]))

        p.setBrush(QBrush(col))
        p.drawEllipse(tip, 3.0, 3.0)

        rect = QRectF(box_lx, box_cy - BOX_H/2, BOX_W, BOX_H)
        path = QPainterPath()
        path.addRoundedRect(rect, 4, 4)
        p.setPen(QPen(col, 1.2)); p.setBrush(QBrush(fill))
        p.drawPath(path)

        p.setFont(QFont("Courier New", 8, QFont.Bold))
        p.setPen(QPen(tc))
        p.drawText(rect.adjusted(6, 0, -4, 0),
                   Qt.AlignVCenter | Qt.AlignLeft, pfx + label)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        # background grid
        p.fillRect(self.rect(), QColor("#0f141d"))
        p.setPen(QPen(QColor(74, 144, 217, 13), 0.5))
        for gx in range(0, W, max(1, W//4)): p.drawLine(gx, 0, gx, H)
        for gy in range(0, H, max(1, H//4)): p.drawLine(0, gy, W, gy)

        # ── layout constants ──────────────────────────────────────
        BOX_W      = 195
        BOX_H      = 22
        BOX_GAP    = 5
        SEP_GAP    = 10
        MARGIN_TOP = 30
        MARGIN_BOT = 14

        n_parts = len(ALL_PARTS)
        n_seps  = len(GROUP_BREAKS)
        total_h = (n_parts * BOX_H
                   + (n_parts - 1) * BOX_GAP
                   + n_seps * SEP_GAP
                   + MARGIN_TOP + MARGIN_BOT)

        col_x   = W - BOX_W - 6
        draw_w  = col_x - 30
        s       = min(draw_w, H * 0.85) / 14.5
        cx      = draw_w * 0.52
        cy      = H * 0.42

        TW, TD  = 10.0, 8.0
        LEG_H   = 4.0
        SHELF_Z = -LEG_H * 0.55

        # ── compute box Y positions upfront ──────────────────────
        start_y = max(MARGIN_TOP, (H - total_h) // 2 + MARGIN_TOP)
        box_centers = []
        y = start_y
        for i, part in enumerate(ALL_PARTS):
            if i in GROUP_BREAKS:
                y += SEP_GAP
            box_centers.append((part, y + BOX_H // 2))
            y += BOX_H + BOX_GAP

        part_box_cy = {part: bcy for part, bcy in box_centers}

        # ── trolley structure ─────────────────────────────────────
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(0,0,0,70)))
        p.drawPolygon(QPolygonF([
            self._iso(0.4,   0.4,   -LEG_H-0.25, cx,cy,s),
            self._iso(TW-0.4,0.4,   -LEG_H-0.25, cx,cy,s),
            self._iso(TW-0.4,TD-0.4,-LEG_H-0.25, cx,cy,s),
            self._iso(0.4,   TD-0.4,-LEG_H-0.25, cx,cy,s),
        ]))

        wc = QColor("#1b1f27"); wp = QPen(QColor("#3a4250"), 1)
        for wx, wy in [(0.5,0.5),(TW-0.5,0.5),(0.5,TD-0.5),(TW-0.5,TD-0.5)]:
            c_ = self._iso(wx, wy, -LEG_H-0.05, cx,cy,s)
            p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(0,0,0,90)))
            p.drawEllipse(c_, s*0.38, s*0.19)
            p.setBrush(QBrush(wc)); p.setPen(wp)
            p.drawEllipse(self._iso(wx,wy,-LEG_H,cx,cy,s), s*0.32, s*0.17)

        lc = QColor("#7d93ab")
        for lx, ly in [(0.5,0.5),(TW-0.5,0.5),(0.5,TD-0.5),(TW-0.5,TD-0.5)]:
            self._post(p, lx, ly, 0.0, -LEG_H, cx,cy,s, w=0.24, col=lc)

        self._box_face(p, 0.3,0.3, TW-0.3,TD-0.3, SHELF_Z-0.3,SHELF_Z,
                       cx,cy,s, QColor("#1b3860"), QColor("#12294a"), QColor("#1a3355"))
        p.setPen(QPen(QColor("#2a5080"), 1.5)); p.setBrush(Qt.NoBrush)
        p.drawPolygon(QPolygonF([
            self._iso(0.3,   0.3,   SHELF_Z,cx,cy,s),
            self._iso(TW-0.3,0.3,   SHELF_Z,cx,cy,s),
            self._iso(TW-0.3,TD-0.3,SHELF_Z,cx,cy,s),
            self._iso(0.3,   TD-0.3,SHELF_Z,cx,cy,s),
        ]))

        self._quad(p, 0.3,0.3, TW-0.3,TD-0.3, -0.05,cx,cy,s,
                   QColor("#2a3f58"), QPen(QColor("#3a6080"),1))

        self._poly(p,
            [self._iso(0,0,-0.5,cx,cy,s), self._iso(0,TD,-0.5,cx,cy,s),
             self._iso(0,TD,0,cx,cy,s),   self._iso(0,0,0,cx,cy,s)],
            QColor("#202b3d"), QPen(QColor("#1a2436"),1))
        self._poly(p,
            [self._iso(0,TD,-0.5,cx,cy,s), self._iso(TW,TD,-0.5,cx,cy,s),
             self._iso(TW,TD,0,cx,cy,s),   self._iso(0,TD,0,cx,cy,s)],
            QColor("#27344a"), QPen(QColor("#1a2436"),1))

        self._quad(p, 0,0, TW,TD, 0,cx,cy,s, QColor("#1d3a5a"), QPen(QColor("#34516f"),2))
        self._quad(p, 0.3,0.3, TW-0.3,TD-0.3, 0,cx,cy,s, QColor("#152c47"), QPen(QColor("#2a4a6c"),1))

        bin_x0 = 7.1
        self._quad(p, bin_x0,0.5, TW-0.4,TD-0.5, 0.01,cx,cy,s,
                   QColor("#11243b"), QPen(QColor("#274867"),1))
        p.setPen(QPen(QColor("#274867"), 0.8))
        for i in range(1, 7):
            yy = 0.5 + i*(TD-1.0)/6.0
            p.drawLine(self._iso(bin_x0,yy,0.02,cx,cy,s), self._iso(TW-0.4,yy,0.02,cx,cy,s))
        mx = (bin_x0+TW-0.4)/2
        p.drawLine(self._iso(mx,0.5,0.02,cx,cy,s), self._iso(mx,TD-0.5,0.02,cx,cy,s))

        # ── blocks on tray (kept for visual structure, no status overlay) ──
        layout = {
            "BLOCK 1": (0.9, 0.7, 2.6, 2.5, 0.6),
            "BLOCK 2": (0.9, 3.7, 2.6, 2.5, 0.6),
            "BLOCK 3": (3.9, 0.7, 2.6, 2.5, 0.6),
            "BLOCK 4": (3.9, 3.7, 2.6, 2.5, 0.6),
        }
        base_colors = {
            "BLOCK 1": (QColor("#0a2040"), QColor("#006070"), QColor("#00bcd4")),
            "BLOCK 2": (QColor("#1a1200"), QColor("#907000"), QColor("#ffd600")),
            "BLOCK 3": (QColor("#200010"), QColor("#801040"), QColor("#e91e8c")),
            "BLOCK 4": (QColor("#0d1a2e"), QColor("#2a4060"), QColor("#7ab0d0")),
        }

        for blk in ("BLOCK 1","BLOCK 3","BLOCK 2","BLOCK 4"):
            gx,gy,gw,gd,h = layout[blk]
            _,c_l,c_t = base_colors[blk]
            self._box_face(p, gx,gy, gx+gw,gy+gd, 0,h,cx,cy,s,
                           c_t.darker(130), c_l.darker(140), c_l.darker(160))

        # overall glow border — based purely on part results
        if self._part_results:
            all_ok = all(v["ok"] for v in self._part_results.values())
            gc = QColor(61,220,132) if all_ok else QColor(255,92,92)
            p.setPen(QPen(gc,3)); p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF([
                self._iso(0, 0, 0,cx,cy,s), self._iso(TW,0, 0,cx,cy,s),
                self._iso(TW,TD,0,cx,cy,s), self._iso(0, TD,0,cx,cy,s),
            ]))

        # handle
        for hx,hy in [(3.0,0.15),(7.0,0.15)]:
            self._post(p,hx,hy,3.0,0.0,cx,cy,s,w=0.18)
        p.setPen(QPen(QColor("#9fb4cb"), max(3,int(s*0.20)), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(self._iso(3.0,0.15,3.0,cx,cy,s), self._iso(7.0,0.15,3.0,cx,cy,s))

        # ── ARROW CALLOUTS ────────────────────────────────────────
        # Arrow tips fan from the right face of each block
        block_anchors = {}
        for blk, (gx,gy,gw,gd,h) in layout.items():
            block_anchors[blk] = self._iso(gx+gw, gy+gd*0.5, h*0.6, cx,cy,s)

        # separators
        p.setPen(QPen(QColor(70,110,165,55), 1, Qt.DashLine))
        for i, part in enumerate(ALL_PARTS):
            if i in GROUP_BREAKS:
                sep_y = part_box_cy[part] - BOX_H//2 - SEP_GAP//2
                p.drawLine(int(col_x), int(sep_y), int(col_x+BOX_W), int(sep_y))

        # one callout per part — tip comes from its block anchor
        for part in ALL_PARTS:
            blk      = PART_TO_BLOCK[part]
            tip      = block_anchors[blk]
            bcy      = part_box_cy[part]
            ok_state = self._part_results.get(part, {}).get("ok", None) if self._part_results else None
            label    = PART_DISPLAY_NAMES.get(part, part)
            self._draw_callout(p, tip, col_x, bcy, label, ok_state)

        # title
        p.setFont(QFont("Courier New", 10, QFont.Bold))
        p.setPen(QPen(QColor("#00e5ff")))
        p.drawText(QRectF(0, 4, draw_w, 22), Qt.AlignHCenter, "KIT TROLLEY · ASSEMBLY LAYOUT")

        p.end()


# ══════════════════════════ FEED PANEL ═════════════════════════════

class FeedPanel(QLabel):
    def __init__(self, placeholder):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:#06080c; color:{C_DIM.name()}; border-radius:8px;")
        self.setText(placeholder)
        self._pix    = None
        self._roi_ok = None

    def set_rgb(self, rgb, roi_ok=None):
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self._pix    = QPixmap.fromImage(img.copy())
        self._roi_ok = roi_ok
        self._rescale()

    def _rescale(self):
        if not self._pix: return
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if self._roi_ok is not None:
            result = QPixmap(scaled)
            qp = QPainter(result)
            qp.setRenderHint(QPainter.Antialiasing)
            sw,sh = result.width(),result.height()
            x1f,y1f,x2f,y2f = ROI_FRAC
            rx1=int(x1f*sw); ry1=int(y1f*sh); rx2=int(x2f*sw); ry2=int(y2f*sh)
            col=QColor(61,220,132) if self._roi_ok else QColor(255,92,92)
            fill=QColor(col); fill.setAlpha(28)
            qp.fillRect(QRect(rx1,ry1,rx2-rx1,ry2-ry1), fill)
            qp.setPen(QPen(col,3)); qp.setBrush(Qt.NoBrush)
            qp.drawRect(QRect(rx1,ry1,rx2-rx1,ry2-ry1))
            bl=18; qp.setPen(QPen(col,4))
            for cx_,cy_,dx,dy in [(rx1,ry1,1,1),(rx2,ry1,-1,1),(rx1,ry2,1,-1),(rx2,ry2,-1,-1)]:
                qp.drawLine(cx_,cy_,cx_+dx*bl,cy_); qp.drawLine(cx_,cy_,cx_,cy_+dy*bl)
            qp.setFont(QFont("Arial",9,QFont.Bold)); qp.setPen(QPen(col))
            qp.drawText(rx1+5, ry1-5, "ROI · OK" if self._roi_ok else "ROI · NOT OK")
            qp.end()
            self.setPixmap(result)
        else:
            self.setPixmap(scaled)

    def resizeEvent(self, e):
        self._rescale(); super().resizeEvent(e)


# ══════════════════════════ HELPERS ════════════════════════════════

def _chip(text, color):
    lbl = QLabel(text)
    lbl.setFont(QFont("Arial", 11, QFont.Bold))
    lbl.setStyleSheet(f"color:{color.name()}; border:1px solid {color.name()}; "
                      "border-radius:6px; padding:4px 10px;")
    return lbl


# ══════════════════════════ MAIN WINDOW ════════════════════════════

class InspectionStation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Kit Inspection")
        self.setStyleSheet(f"background:{C_BG};")
        self.image_mode     = START_IN_IMAGE_MODE
        self.countdown      = 0
        self._pending_image = None

        title = QLabel("SMART KIT INSPECTION")
        title.setFont(QFont("Arial", 19, QFont.Bold))
        title.setStyleSheet(f"color:{C_TEXT}; letter-spacing:2px;")
        sub = QLabel("Part Verification · ZONE 2")
        sub.setStyleSheet(f"color:{C_DIM.name()};")
        hl = QVBoxLayout(); hl.addWidget(title); hl.addWidget(sub)

        self.mode_lbl = QLabel("● IMAGE TEST MODE")
        self.mode_lbl.setStyleSheet("color:#f0a83d; font-weight:bold;")

        self.logo = TataMotorsLogo()
        self.logo.setFixedSize(500, 140)

        header = QHBoxLayout()
        header.addLayout(hl); header.addStretch()
        header.addWidget(self.mode_lbl); header.addSpacing(14); header.addWidget(self.logo)

        self.trolley = TrolleyView()
        lt = QLabel("REFERENCE TROLLEY · 3D VIEW")
        lt.setFont(QFont("Arial", 11, QFont.Bold))
        lt.setStyleSheet(f"color:{C_DIM.name()}; letter-spacing:2px;")
        left = QVBoxLayout(); left.addWidget(lt); left.addWidget(self.trolley,1)
        left_frame = self._wrap(left)

        rt_title = QLabel("LIVE CAMERA · SOURCE FEED")
        rt_title.setFont(QFont("Arial", 11, QFont.Bold))
        rt_title.setStyleSheet(f"color:{C_DIM.name()}; letter-spacing:2px;")

        self.add_btn  = QPushButton("+  Add Image")
        self.scan_btn = QPushButton("Scan Now")
        self.live_btn = QPushButton("Use Live Feed")
        self._style_btn(self.add_btn,"#2d6cdf")
        self._style_btn(self.scan_btn,"#3a3f4b")
        self._style_btn(self.live_btn,"#3a3f4b")
        self.add_btn.clicked.connect(self.on_add_image)
        self.scan_btn.clicked.connect(self.trigger_scan)
        self.live_btn.clicked.connect(self.toggle_mode)

        rt = QHBoxLayout()
        rt.addWidget(rt_title); rt.addStretch()
        rt.addWidget(self.add_btn); rt.addWidget(self.scan_btn); rt.addWidget(self.live_btn)

        self.feed = FeedPanel("Add an image to test the model,\nor switch to the live feed.")
        right = QVBoxLayout(); right.addLayout(rt); right.addWidget(self.feed,1)
        right_frame = self._wrap(right)

        panels = QHBoxLayout()
        panels.addWidget(left_frame,1); panels.addSpacing(12); panels.addWidget(right_frame,1)

        self.bar_title = QLabel("LIVE KIT VERIFICATION")
        self.bar_title.setFont(QFont("Arial",12,QFont.Bold))
        self.bar_title.setStyleSheet(f"color:{C_DIM.name()}; letter-spacing:2px;")

        chip_row = QHBoxLayout()
        chip_row.addWidget(self.bar_title); chip_row.addSpacing(10)
        self.verdict_chip = _chip("WAITING", C_DIM)
        self.verdict_chip.setFont(QFont("Arial",13,QFont.Bold))
        chip_row.addWidget(self.verdict_chip); chip_row.addStretch()
        self.status_lbl = QLabel("Loading model…")
        self.status_lbl.setFont(QFont("Arial",13))
        self.status_lbl.setStyleSheet(f"color:{C_TEXT}; font-weight:bold;")
        chip_row.addWidget(self.status_lbl)

        self.progress = QProgressBar()
        self.progress.setRange(0,100); self.progress.setValue(0)
        self.progress.setTextVisible(False); self.progress.setFixedHeight(8)
        self.progress.setStyleSheet(
            "QProgressBar{background:#222833;border:none;border-radius:4px;}"
            "QProgressBar::chunk{background:#2d6cdf;border-radius:4px;}")

        bar = QVBoxLayout(); bar.addLayout(chip_row); bar.addWidget(self.progress)
        bar_frame = self._wrap(bar); bar_frame.setMaximumHeight(110)

        root = QVBoxLayout(self)
        root.setContentsMargins(18,14,18,14)
        root.addLayout(header); root.addSpacing(8)
        root.addLayout(panels,1); root.addSpacing(10)
        root.addWidget(bar_frame)

        self.worker = ScanWorker()
        self.worker.model_ready.connect(self.on_model_ready)
        self.worker.scan_result.connect(self.on_scan_result)
        self.worker.scan_failed.connect(self.on_scan_failed)
        self.worker.start()

        self.timer = QTimer(self); self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self.busy = QTimer(self); self.busy.setInterval(60)
        self._busy_val = 0
        self.busy.timeout.connect(self._busy_pulse)
        self._apply_mode_ui()

    def _wrap(self, layout):
        f = QFrame()
        f.setStyleSheet(f"QFrame{{background:{C_PANEL}; border-radius:12px;}}")
        f.setLayout(layout); layout.setContentsMargins(16,12,16,12)
        return f

    def _style_btn(self, b, bg):
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(
            f"QPushButton{{background:{bg}; color:white; border:none; border-radius:7px;"
            " padding:7px 14px; font-weight:bold;}}"
            "QPushButton:disabled{background:#2a2f3a; color:#5c6470;}")

    def on_model_ready(self, ok, msg):
        if ok:
            self.status_lbl.setText("Ready")
            self.status_lbl.setStyleSheet(f"color:{C_OK.name()}; font-weight:bold;")
            if not self.image_mode: self.trigger_scan()
        else:
            self.status_lbl.setText(msg)
            self.status_lbl.setStyleSheet(f"color:{C_BAD.name()}; font-weight:bold;")

    def trigger_scan(self):
        if self.image_mode and self._pending_image is not None:
            self._start_busy("SCANNING…")
            self.worker.request_image_scan(self._pending_image)
        elif not self.image_mode:
            self._start_busy("SCANNING…")
            self.worker.request_camera_scan()
        else:
            self.status_lbl.setText("Add an image first")
            self.status_lbl.setStyleSheet("color:#f0a83d; font-weight:bold;")

    def on_scan_result(self, rgb, part_results, any_not_ok):
        self._stop_busy()
        verdict_ok = not any_not_ok
        self.feed.set_rgb(rgb, roi_ok=verdict_ok)
        self.trolley.set_state(part_results)

        ok_count     = sum(1 for p in part_results.values() if p["ok"])
        not_ok_count = len(part_results) - ok_count
        vcol = C_OK if verdict_ok else C_BAD

        self.verdict_chip.setText(f"✓ OK : {ok_count}    ✕ NOT OK : {not_ok_count}")
        self.verdict_chip.setStyleSheet(
        f"color:{vcol.name()}; border:1px solid {vcol.name()}; "
        "border-radius:6px; padding:4px 12px; font-weight:bold;")

        if self.image_mode:
            self.status_lbl.setText(f"Detected {ok_count}/{len(part_results)} parts")
            self.status_lbl.setStyleSheet(f"color:{vcol.name()}; font-weight:bold;")
        else:
            self.countdown = RESCAN_SECONDS
            self.timer.start()
            self._tick()

    def on_scan_failed(self, msg):
        self._stop_busy()
        self.status_lbl.setText(f"Scan failed: {msg}")
        self.status_lbl.setStyleSheet(f"color:{C_BAD.name()}; font-weight:bold;")
        if not self.image_mode:
            self.countdown = RESCAN_SECONDS; self.timer.start(); self._tick()

    def _tick(self):
        if self.countdown <= 0:
            self.timer.stop(); self.trigger_scan(); return
        self.status_lbl.setText(f"Next scan in {self.countdown}s")
        self.status_lbl.setStyleSheet(f"color:{C_TEXT}; font-weight:bold;")
        self.progress.setValue(int(100*(RESCAN_SECONDS-self.countdown)/RESCAN_SECONDS))
        self.countdown -= 1

    def _start_busy(self, text):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet("color:#f0a83d; font-weight:bold;")
        self._busy_val = 0; self.busy.start()
        self.scan_btn.setEnabled(False); self.add_btn.setEnabled(False)

    def _busy_pulse(self):
        self._busy_val = (self._busy_val+4) % 100
        self.progress.setValue(self._busy_val)

    def _stop_busy(self):
        self.busy.stop()
        self.scan_btn.setEnabled(True); self.add_btn.setEnabled(True)

    def on_add_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"Select test image","",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path: return
        frame = cv2.imread(path)
        if frame is None:
            self.status_lbl.setText("Could not read image")
            self.status_lbl.setStyleSheet(f"color:{C_BAD.name()}; font-weight:bold;"); return
        self._pending_image = frame
        if not self.image_mode:
            self.image_mode = True; self._apply_mode_ui()
        self.feed.set_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.trigger_scan()

    def toggle_mode(self):
        self.image_mode = not self.image_mode
        self._apply_mode_ui()
        if not self.image_mode: self.trigger_scan()

    def _apply_mode_ui(self):
        if self.image_mode:
            self.mode_lbl.setText("● IMAGE TEST MODE")
            self.mode_lbl.setStyleSheet("color:#f0a83d; font-weight:bold;")
            self.live_btn.setText("Use Live Feed")
            self.timer.stop(); self.progress.setValue(0)
        else:
            self.mode_lbl.setText("● LIVE FEED")
            self.mode_lbl.setStyleSheet(f"color:{C_OK.name()}; font-weight:bold;")
            self.live_btn.setText("Image Test")
        self.scan_btn.setVisible(self.image_mode)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_Q): self.close()

    def closeEvent(self, e):
        self.worker.stop(); super().closeEvent(e)


# ══════════════════════════ ENTRY POINT ════════════════════════════

def main():
    app = QApplication(sys.argv)
    win = InspectionStation()
    if FULLSCREEN:
        win.showFullScreen()
    else:
        win.resize(1380, 880); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
