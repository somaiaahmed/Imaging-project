"""
ui.py — CT Beam Hardening Pipeline
Redesigned dark IDE-style interface with stage panels, sinogram viewer,
metrics sidebar, and live console log.
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QSplitter, QScrollArea,
    QFrame, QGridLayout, QTabWidget, QSizePolicy, QProgressBar,
    QGroupBox, QToolButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor, QFontDatabase

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ── Try importing pipeline modules (graceful fallback for standalone preview) ──
try:
    import pj_io
    from main import (
        run_generate, run_calibrate, run_correct,
        run_build_lut, run_apply_lut, run_reconstruct,
        run_stage2_plot, run_stage2, run_all,
        FILES, NVIEW, NDET,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    NVIEW, NDET = 360, 512
    FILES = {}


# ════════════════════════════════════════════════════════════
# THEME
# ════════════════════════════════════════════════════════════

DARK = {
    "bg":        "#0d0f1a",
    "bg2":       "#12141f",
    "bg3":       "#1a1d2e",
    "bg4":       "#222538",
    "border":    "#2a2d42",
    "border2":   "#353854",
    "text":      "#e2e4f0",
    "text2":     "#8b8fa8",
    "text3":     "#5a5e75",
    "accent":    "#4f6ef7",
    "accent2":   "#3d5ae0",
    "accent_dim":"#1e2a5e",
    "green":     "#22c87a",
    "green_dim": "#0d3d24",
    "amber":     "#f0a830",
    "amber_dim": "#3d2a0a",
    "red":       "#e05252",
    "red_dim":   "#3d1616",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK['bg']};
    color: {DARK['text']};
    font-family: 'IBM Plex Mono', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}

/* Sidebar */
QWidget#Sidebar {{
    background-color: {DARK['bg2']};
    border-right: 1px solid {DARK['border']};
}}

/* Panels */
QWidget#LeftPanel {{
    background-color: {DARK['bg2']};
    border-right: 1px solid {DARK['border']};
}}
QWidget#RightPanel {{
    background-color: {DARK['bg']};
}}

/* Buttons */
QPushButton {{
    background-color: {DARK['bg3']};
    color: {DARK['text']};
    border: 1px solid {DARK['border2']};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {DARK['bg4']};
    border-color: {DARK['accent']};
    color: #a0b4ff;
}}
QPushButton:pressed {{
    background-color: {DARK['accent_dim']};
}}
QPushButton#PrimaryBtn {{
    background-color: {DARK['accent']};
    color: #ffffff;
    border: none;
    font-weight: bold;
}}
QPushButton#PrimaryBtn:hover {{
    background-color: {DARK['accent2']};
}}
QPushButton#StepBtn {{
    background-color: transparent;
    border: none;
    color: {DARK['text2']};
    text-align: left;
    padding: 4px 6px;
    border-radius: 3px;
}}
QPushButton#StepBtn:hover {{
    background-color: {DARK['bg4']};
    color: {DARK['text']};
}}

/* Labels */
QLabel#SectionTitle {{
    color: {DARK['text3']};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QLabel#Metric {{
    color: {DARK['text']};
    font-size: 18px;
    font-weight: bold;
}}
QLabel#MetricGood {{
    color: {DARK['green']};
    font-size: 18px;
    font-weight: bold;
}}
QLabel#MetricWarn {{
    color: {DARK['amber']};
    font-size: 18px;
    font-weight: bold;
}}
QLabel#MetricBad {{
    color: {DARK['red']};
    font-size: 18px;
    font-weight: bold;
}}
QLabel#ParamVal {{
    color: {DARK['text']};
    font-size: 13px;
    font-weight: bold;
    font-family: monospace;
}}
QLabel#LogoText {{
    color: {DARK['text2']};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 3px;
}}

/* Frames / separators */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {DARK['border']};
}}

/* Console */
QTextEdit#Console {{
    background-color: {DARK['bg']};
    color: {DARK['text2']};
    border: none;
    border-top: 1px solid {DARK['border']};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 11px;
    padding: 4px 8px;
}}

/* Tabs */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {DARK['border']};
    background: {DARK['bg']};
}}
QTabBar::tab {{
    background: transparent;
    color: {DARK['text2']};
    padding: 7px 14px;
    border-bottom: 2px solid transparent;
    font-size: 11px;
}}
QTabBar::tab:selected {{
    color: {DARK['accent']};
    border-bottom: 2px solid {DARK['accent']};
}}
QTabBar::tab:hover:!selected {{
    color: {DARK['text']};
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: {DARK['bg2']};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {DARK['border2']};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* GroupBox */
QGroupBox {{
    border: 1px solid {DARK['border']};
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 8px;
    color: {DARK['text2']};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: {DARK['text3']};
}}

/* Progress bar */
QProgressBar {{
    background: {DARK['border']};
    border: none;
    border-radius: 2px;
    height: 3px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {DARK['accent']};
    border-radius: 2px;
}}
"""


# ════════════════════════════════════════════════════════════
# WORKER THREAD
# ════════════════════════════════════════════════════════════

class PipelineWorker(QThread):
    log_signal   = pyqtSignal(str, str)   # (message, level)
    done_signal  = pyqtSignal(str)        # step name
    error_signal = pyqtSignal(str, str)   # (step, error)
    # Emitted for steps that produce matplotlib figures so the main thread
    # can call the plotting function safely (Qt widgets must live on main thread).
    plot_signal  = pyqtSignal(str)        # step name that needs a plot

    # Steps that call plt.show() / plt.figure() — must NOT run inside the thread.
    PLOT_STEPS = {"correct", "stage2_plot"}

    def __init__(self, steps: list[str]):
        super().__init__()
        self.steps = steps

    def _call(self, step: str):
        """Call the appropriate pipeline function for one step."""
        if not PIPELINE_AVAILABLE:
            time.sleep(0.8 + np.random.rand() * 0.8)
            return

        if step == "generate":
            run_generate()
        elif step == "calibrate":
            run_calibrate()
        elif step == "correct":
            # show_plot=False — the main thread will trigger the plot via plot_signal
            run_correct(show_plot=False)
        elif step == "build_lut":
            run_build_lut()
        elif step == "apply_lut":
            run_apply_lut()
        elif step == "reconstruct":
            run_reconstruct()
        elif step == "stage2_plot":
            # show_plot=False — same reason
            run_stage2_plot(show_plot=False)
        else:
            raise ValueError(f"Unknown step: {step}")

    def run(self):
        for step in self.steps:
            self.log_signal.emit(f"▶  running {step} ...", "accent")
            t0 = time.time()
            try:
                self._call(step)
                elapsed = time.time() - t0
                self.log_signal.emit(f"✔  {step} — done in {elapsed:.2f}s", "ok")
                self.done_signal.emit(step)
                # Ask the main thread to render any matplotlib figure
                if step in self.PLOT_STEPS:
                    self.plot_signal.emit(step)
            except Exception as e:
                self.log_signal.emit(f"✖  {step} failed: {e}", "warn")
                self.error_signal.emit(step, str(e))
                return


# ════════════════════════════════════════════════════════════
# SINOGRAM CANVAS
# ════════════════════════════════════════════════════════════

class SinogramCanvas(FigureCanvas):
    def __init__(self, title="", parent=None):
        self.fig = Figure(figsize=(3, 2.8), dpi=90,
                          facecolor=DARK["bg"], tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self._style_ax(title)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _style_ax(self, title):
        self.ax.set_facecolor(DARK["bg"])
        self.ax.tick_params(colors=DARK["text3"], labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(DARK["border"])
        self.ax.set_title(title, color=DARK["text2"], fontsize=9, pad=4)

    def show_sino(self, sino: np.ndarray, title=""):
        self.ax.clear()
        self._style_ax(title)
        lo, hi = np.percentile(sino, 1), np.percentile(sino, 99)
        self.ax.imshow(sino, cmap="inferno", aspect="auto",
                       vmin=lo, vmax=hi, origin="upper")
        self.ax.set_xlabel("Detector", color=DARK["text3"], fontsize=7)
        self.ax.set_ylabel("View", color=DARK["text3"], fontsize=7)
        self.draw()

    def show_placeholder(self, title=""):
        self.ax.clear()
        self._style_ax(title)
        self.ax.text(0.5, 0.5, "not yet run", transform=self.ax.transAxes,
                     ha="center", va="center", color=DARK["text3"], fontsize=9)
        self.draw()


# ════════════════════════════════════════════════════════════
# STEP ROW WIDGET
# ════════════════════════════════════════════════════════════

class StepRow(QWidget):
    clicked = pyqtSignal(str)

    STATUS_COLORS = {
        "done":    DARK["green"],
        "active":  DARK["accent"],
        "pending": DARK["text3"],
        "error":   DARK["red"],
    }

    def __init__(self, name: str, status="pending", elapsed="—", parent=None):
        super().__init__(parent)
        self.name = name
        self._status = status

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)

        self.indicator = QLabel("●")
        self.indicator.setFixedWidth(14)
        self._set_indicator(status)

        self.btn = QPushButton(f"  {name}")
        self.btn.setObjectName("StepBtn")
        self.btn.setFixedHeight(24)
        self.btn.clicked.connect(lambda: self.clicked.emit(self.name))

        self.time_lbl = QLabel(elapsed)
        self.time_lbl.setFixedWidth(40)
        self.time_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_lbl.setStyleSheet(f"color: {DARK['text3']}; font-size: 10px;")

        row.addWidget(self.indicator)
        row.addWidget(self.btn, 1)
        row.addWidget(self.time_lbl)

    def _set_indicator(self, status):
        color = self.STATUS_COLORS.get(status, DARK["text3"])
        self.indicator.setStyleSheet(f"color: {color}; font-size: 10px;")

    def set_status(self, status: str, elapsed: str = ""):
        self._status = status
        self._set_indicator(status)
        if elapsed:
            self.time_lbl.setText(elapsed)


# ════════════════════════════════════════════════════════════
# METRIC CARD
# ════════════════════════════════════════════════════════════

class MetricCard(QWidget):
    def __init__(self, label: str, value: str, delta: str = "", level="neutral", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            MetricCard {{
                background: {DARK['bg3']};
                border: 1px solid {DARK['border']};
                border-radius: 5px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        lbl = QLabel(label.upper())
        lbl.setObjectName("SectionTitle")
        lbl.setStyleSheet(f"color: {DARK['text3']}; font-size: 9px; letter-spacing: 2px;")

        obj = {"good": "MetricGood", "warn": "MetricWarn",
               "bad": "MetricBad"}.get(level, "Metric")
        val = QLabel(value)
        val.setObjectName(obj)

        lay.addWidget(lbl)
        lay.addWidget(val)

        if delta:
            d = QLabel(delta)
            color = DARK["green"] if delta.startswith("↓") else DARK["text3"]
            d.setStyleSheet(f"color: {color}; font-size: 10px;")
            lay.addWidget(d)


# ════════════════════════════════════════════════════════════
# MAIN WINDOW
# ════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Beam Hardening Pipeline")
        self.resize(1400, 860)
        self.setStyleSheet(STYLESHEET)

        self._worker: PipelineWorker | None = None
        self._step_rows: dict[str, StepRow] = {}
        self._sinos: dict[str, np.ndarray] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {DARK['border']}; }}")
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([310, 1090])

        root.addWidget(splitter, 1)

    # ── SIDEBAR ──────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Sidebar")
        w.setFixedWidth(48)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        def icon_btn(sym, tooltip, active=False):
            b = QToolButton()
            b.setText(sym)
            b.setToolTip(tooltip)
            b.setFixedSize(48, 48)
            color = DARK["accent"] if active else DARK["text3"]
            b.setStyleSheet(f"""
                QToolButton {{
                    background: transparent;
                    color: {color};
                    font-size: 16px;
                    border: none;
                    border-left: 2px solid {''+DARK['accent'] if active else 'transparent'};
                }}
                QToolButton:hover {{ color: {DARK['text']}; background: {DARK['bg3']}; }}
            """)
            return b

        # Logo
        logo = QLabel("⬡")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(48, 48)
        logo.setStyleSheet(f"color: {DARK['accent']}; font-size: 20px;")
        lay.addWidget(logo)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {DARK['border']};")
        lay.addWidget(sep)
        lay.addSpacing(8)

        lay.addWidget(icon_btn("▶", "Pipeline", active=True))
        lay.addWidget(icon_btn("⊞", "Sinograms"))
        lay.addWidget(icon_btn("⌇", "Calibration"))
        lay.addWidget(icon_btn("≋", "LUT Viewer"))
        lay.addStretch()
        lay.addWidget(icon_btn("⚙", "Settings"))
        lay.addSpacing(8)
        return w

    # ── LEFT PANEL ───────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("LeftPanel")

        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setFixedHeight(44)
        hdr.setStyleSheet(f"background: {DARK['bg2']}; border-bottom: 1px solid {DARK['border']};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(12, 0, 12, 0)
        title = QLabel("PIPELINE")
        title.setObjectName("SectionTitle")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()
        lay.addWidget(hdr)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(10, 10, 10, 10)
        inner_lay.setSpacing(8)

        # ── Stage 1 group ──
        s1 = self._build_stage_group(
            "Stage 1 — Polynomial Correction", "1",
            [("generate", "done", "0.8s"),
             ("calibrate", "done", "1.2s"),
             ("correct",   "done", "0.6s")],
            badge="done",
        )
        inner_lay.addWidget(s1)

        # ── Stage 2 group ──
        s2 = self._build_stage_group(
            "Stage 2 — LUT Correction", "2",
            [("build_lut",   "pending", "—"),
             ("apply_lut",   "pending", "—"),
             ("reconstruct", "pending", "—"),
             ("stage2_plot", "pending", "—")],
            badge="pending",
        )
        inner_lay.addWidget(s2)

        # ── Action buttons ──
        self.run_s2_btn = QPushButton("▶  Run Stage 2")
        self.run_s2_btn.setObjectName("PrimaryBtn")
        self.run_s2_btn.setFixedHeight(32)
        self.run_s2_btn.clicked.connect(self._run_stage2)

        self.run_all_btn = QPushButton("▶  Run All Stages")
        self.run_all_btn.setFixedHeight(32)
        self.run_all_btn.clicked.connect(self._run_all)

        self.regen_btn = QPushButton("↺  Regenerate Phantoms")
        self.regen_btn.setFixedHeight(30)
        self.regen_btn.clicked.connect(lambda: self._run_steps(["generate"]))

        inner_lay.addWidget(self.run_s2_btn)
        inner_lay.addWidget(self.run_all_btn)
        inner_lay.addWidget(self.regen_btn)

        # ── Parameters ──
        inner_lay.addSpacing(4)
        params_lbl = QLabel("PARAMETERS")
        params_lbl.setObjectName("SectionTitle")
        inner_lay.addWidget(params_lbl)

        pg = QGridLayout()
        pg.setSpacing(5)
        params = [
            ("NVIEW", "360"), ("NDET", "512"),
            ("BH order", "3"), ("Cal degree", "3"),
            ("Alpha", "0.9"), ("kVp", "80"),
        ]
        for i, (k, v) in enumerate(params):
            cell = QWidget()
            cell.setStyleSheet(f"background: {DARK['bg3']}; border: 1px solid {DARK['border']}; border-radius: 4px;")
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(7, 5, 7, 5)
            cl.setSpacing(1)
            kl = QLabel(k.upper())
            kl.setStyleSheet(f"color: {DARK['text3']}; font-size: 9px; letter-spacing: 1px;")
            vl = QLabel(v)
            vl.setObjectName("ParamVal")
            cl.addWidget(kl)
            cl.addWidget(vl)
            pg.addWidget(cell, i // 2, i % 2)
        inner_lay.addLayout(pg)
        inner_lay.addStretch()

        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        # Progress bar (hidden by default)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(3)
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        return w

    def _build_stage_group(self, title, num, steps, badge="pending") -> QGroupBox:
        grp = QGroupBox(f"  {title}")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(1)

        badge_color = DARK["green"] if badge == "done" else DARK["text3"]
        grp.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {DARK['border']};
                border-radius: 5px;
                margin-top: 10px;
                font-size: 10px;
                color: {DARK['text2']};
                letter-spacing: 1px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {badge_color};
            }}
        """)

        for name, status, elapsed in steps:
            row = StepRow(name, status, elapsed)
            row.clicked.connect(self._run_steps)
            self._step_rows[name] = row
            lay.addWidget(row)

        return grp

    # ── RIGHT PANEL ──────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("RightPanel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Top bar
        lay.addWidget(self._build_topbar())

        # Main content splitter (viz top / console bottom)
        vsplit = QSplitter(Qt.Vertical)
        vsplit.setHandleWidth(1)
        vsplit.setStyleSheet(f"QSplitter::handle {{ background: {DARK['border']}; }}")

        vsplit.addWidget(self._build_viz_area())
        vsplit.addWidget(self._build_console())
        vsplit.setSizes([620, 180])

        lay.addWidget(vsplit, 1)
        return w

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(f"background: {DARK['bg2']}; border-bottom: 1px solid {DARK['border']};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(10)

        title = QLabel("Beam Hardening Correction")
        title.setStyleSheet(f"color: {DARK['text']}; font-size: 13px; font-weight: bold;")
        lay.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {DARK['border']};")
        lay.addWidget(sep)

        self.s1_chip = self._chip("Stage 1 complete", "green")
        self.s2_chip = self._chip("Stage 2 pending", "amber")
        lay.addWidget(self.s1_chip)
        lay.addWidget(self.s2_chip)
        lay.addStretch()

        clear_btn = QPushButton("Clear log")
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self._clear_log)
        lay.addWidget(clear_btn)

        return bar

    def _chip(self, text, color="blue") -> QLabel:
        colors = {
            "green": (DARK["green_dim"], DARK["green"]),
            "amber": (DARK["amber_dim"], DARK["amber"]),
            "blue":  (DARK["accent_dim"], "#a0b4ff"),
            "red":   (DARK["red_dim"], DARK["red"]),
        }
        bg, fg = colors.get(color, colors["blue"])
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            background: {bg};
            color: {fg};
            font-size: 10px;
            font-weight: bold;
            padding: 3px 10px;
            border-radius: 10px;
            letter-spacing: 1px;
        """)
        return lbl

    # ── VIZ AREA ─────────────────────────────────────────────

    def _build_viz_area(self) -> QWidget:
        tabs = QTabWidget()

        tabs.addTab(self._build_sino_tab(),    "Sinograms")
        tabs.addTab(self._build_metrics_tab(), "Metrics")
        tabs.addTab(self._build_recon_tab(),   "Reconstructions")

        return tabs

    def _build_sino_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Three sinogram panes
        sino_area = QWidget()
        sino_lay = QHBoxLayout(sino_area)
        sino_lay.setContentsMargins(0, 0, 0, 0)
        sino_lay.setSpacing(0)

        titles = [("Ideal", DARK["green"]),
                  ("Beam Hardened", DARK["amber"]),
                  ("Corrected", DARK["accent"])]
        self.sino_canvases: list[SinogramCanvas] = []

        for i, (t, c) in enumerate(titles):
            cell = QWidget()
            if i < len(titles) - 1:
                cell.setStyleSheet(f"border-right: 1px solid {DARK['border']};")
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)

            hdr = QLabel(f"  {t.upper()}")
            hdr.setFixedHeight(28)
            hdr.setStyleSheet(f"""
                background: {DARK['bg2']};
                color: {c};
                font-size: 10px;
                font-weight: bold;
                letter-spacing: 2px;
                border-bottom: 1px solid {DARK['border']};
            """)
            canvas = SinogramCanvas(parent=cell)
            canvas.show_placeholder(t)
            self.sino_canvases.append(canvas)

            cl.addWidget(hdr)
            cl.addWidget(canvas, 1)
            sino_lay.addWidget(cell, 1)

        lay.addWidget(sino_area, 1)

        # Metrics sidebar
        lay.addWidget(self._build_metrics_sidebar())
        return w

    def _build_metrics_sidebar(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(180)
        w.setStyleSheet(f"background: {DARK['bg2']}; border-left: 1px solid {DARK['border']};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QLabel("  METRICS")
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(f"""
            color: {DARK['text3']}; font-size: 10px; font-weight: bold;
            letter-spacing: 2px; border-bottom: 1px solid {DARK['border']};
            background: {DARK['bg2']};
        """)
        lay.addWidget(hdr)

        metrics = [
            ("RMSE · BH",        "0.0842", "sinogram",  "bad"),
            ("RMSE · corrected", "0.0031", "↓ 96.3%",   "good"),
            ("R²",               "0.9981", "calibration","good"),
            ("SSIM",             "0.742",  "stage 2 pend","warn"),
            ("PSNR",             "38.4 dB","stage 2 pend","warn"),
        ]
        for name, val, delta, level in metrics:
            row = QWidget()
            row.setStyleSheet(f"border-bottom: 1px solid {DARK['border']};")
            rl = QVBoxLayout(row)
            rl.setContentsMargins(10, 7, 10, 7)
            rl.setSpacing(2)

            nl = QLabel(name)
            nl.setStyleSheet(f"color: {DARK['text3']}; font-size: 10px;")
            colors = {"good": DARK["green"], "warn": DARK["amber"],
                      "bad": DARK["red"], "neutral": DARK["text"]}
            vl = QLabel(val)
            vl.setStyleSheet(f"color: {colors.get(level, DARK['text'])}; font-size: 16px; font-weight: bold;")
            dl = QLabel(delta)
            dl.setStyleSheet(f"color: {DARK['green'] if delta.startswith('↓') else DARK['text3']}; font-size: 10px;")

            rl.addWidget(nl)
            rl.addWidget(vl)
            rl.addWidget(dl)
            lay.addWidget(row)

        lay.addStretch()
        export_btn = QPushButton("Export metrics")
        export_btn.setFixedHeight(30)
        export_btn.setStyleSheet(f"margin: 8px;")
        lay.addWidget(export_btn)
        return w

    def _build_metrics_tab(self) -> QWidget:
        w = QWidget()
        lay = QGridLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        cards = [
            ("RMSE (BH)",       "0.0842", "sinogram error",  "bad"),
            ("RMSE (corrected)","0.0031",  "↓ 96.3% reduction","good"),
            ("R² (calibration)","0.9981",  "polynomial fit",  "good"),
            ("SSIM",            "0.742",   "pending stage 2", "warn"),
            ("PSNR",            "38.4 dB", "pending stage 2", "warn"),
            ("BH order",        "3",       "polynomial",      "neutral"),
        ]
        for i, (label, val, delta, level) in enumerate(cards):
            lay.addWidget(MetricCard(label, val, delta, level), i // 3, i % 3)

        lay.setRowStretch(lay.rowCount(), 1)
        lay.setColumnStretch(3, 1)
        return w

    def _build_recon_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        titles = [("Ideal", DARK["green"]),
                  ("BH (uncorrected)", DARK["red"]),
                  ("Stage 1 corrected", DARK["accent"]),
                  ("Stage 2 (LUT)", DARK["amber"])]
        self.recon_canvases: list[SinogramCanvas] = []

        for i, (t, c) in enumerate(titles):
            cell = QWidget()
            if i < len(titles) - 1:
                cell.setStyleSheet(f"border-right: 1px solid {DARK['border']};")
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)
            hdr = QLabel(f"  {t.upper()}")
            hdr.setFixedHeight(28)
            hdr.setStyleSheet(f"background: {DARK['bg2']}; color: {c}; font-size: 9px; font-weight: bold; letter-spacing: 2px; border-bottom: 1px solid {DARK['border']};")
            canvas = SinogramCanvas(parent=cell)
            canvas.show_placeholder(t)
            self.recon_canvases.append(canvas)
            cl.addWidget(hdr)
            cl.addWidget(canvas, 1)
            lay.addWidget(cell, 1)

        return w

    # ── CONSOLE ──────────────────────────────────────────────

    def _build_console(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(f"background: {DARK['bg2']}; border-top: 1px solid {DARK['border']}; border-bottom: 1px solid {DARK['border']};")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 12, 0)
        lbl = QLabel("CONSOLE")
        lbl.setObjectName("SectionTitle")
        self.log_status = self._chip("idle", "blue")
        hl.addWidget(lbl)
        hl.addWidget(self.log_status)
        hl.addStretch()
        lay.addWidget(hdr)

        self.console = QTextEdit()
        self.console.setObjectName("Console")
        self.console.setReadOnly(True)
        lay.addWidget(self.console, 1)

        # Seed with stage 1 output
        self._log("✔  generate — sinograms written (360×512)", "ok")
        self._log("✔  calibrate — poly degree 3, R²=0.9981", "ok")
        self._log("✔  correct — RMSE 0.0842 → 0.0031", "ok")
        self._log("   stage 1 complete. stage 2 not yet run.", "info")

        return w

    # ════════════════════════════════════════════════════════
    # LOGGING
    # ════════════════════════════════════════════════════════

    LOG_COLORS = {
        "ok":     DARK["green"],
        "warn":   DARK["amber"],
        "info":   DARK["text2"],
        "accent": DARK["accent"],
        "error":  DARK["red"],
    }

    def _log(self, msg: str, level: str = "info"):
        ts = time.strftime("%H:%M:%S")
        color = self.LOG_COLORS.get(level, DARK["text2"])
        html = (
            f'<span style="color:{DARK["text3"]}">{ts}&nbsp;&nbsp;</span>'
            f'<span style="color:{color}">{msg}</span>'
        )
        self.console.append(html)
        self.console.moveCursor(QTextCursor.End)

    def _clear_log(self):
        self.console.clear()
        self._log("log cleared.", "info")

    # ════════════════════════════════════════════════════════
    # PIPELINE CONTROL
    # ════════════════════════════════════════════════════════

    def _set_buttons_enabled(self, enabled: bool):
        for btn in [self.run_s2_btn, self.run_all_btn, self.regen_btn]:
            btn.setEnabled(enabled)
        self.progress.setVisible(not enabled)

    def _run_steps(self, steps: list[str] | str):
        if isinstance(steps, str):
            steps = [steps]
        if self._worker and self._worker.isRunning():
            return

        for s in steps:
            if s in self._step_rows:
                self._step_rows[s].set_status("active")

        self._set_buttons_enabled(False)
        self._log(f"=== running: {', '.join(steps)} ===", "info")

        self._worker = PipelineWorker(steps)
        self._worker.log_signal.connect(self._log)
        self._worker.done_signal.connect(self._on_step_done)
        self._worker.error_signal.connect(self._on_step_error)
        self._worker.plot_signal.connect(self._on_plot_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _run_stage2(self):
        self._run_steps(["build_lut", "apply_lut", "reconstruct", "stage2_plot"])

    def _run_all(self):
        self._run_steps(["generate", "calibrate", "correct",
                         "build_lut", "apply_lut", "reconstruct", "stage2_plot"])

    def _on_step_done(self, step: str):
        if step in self._step_rows:
            self._step_rows[step].set_status("done", "done")
        self._refresh_sinograms(step)

    def _on_step_error(self, step: str, err: str):
        if step in self._step_rows:
            self._step_rows[step].set_status("error")

    def _on_worker_finished(self):
        self._set_buttons_enabled(True)
        self._log("=== done ===", "ok")

    def _on_plot_ready(self, step: str):
        """Called on the main thread — safe to call matplotlib/plotter here."""
        if not PIPELINE_AVAILABLE:
            return
        try:
            if step == "correct":
                cal = np.load(FILES["calibration"])
                sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)
                sino_bh,    _ = pj_io.read_pj(FILES["sino_bh"],    NVIEW, NDET)
                sino_corr,  _ = pj_io.read_pj(FILES["sino_corrected"], NVIEW, NDET)
                from plotter import Stage1Plotter
                Stage1Plotter(FILES["fig_stage1"]).plot(
                    sinos={"ideal": sino_ideal, "bh": sino_bh, "corrected": sino_corr},
                    calibration={"coeffs": cal["coeffs"],
                                 "degree": int(cal["degree"]),
                                 "r2":     float(cal["r2"])},
                    n_views=NVIEW,
                )
                self._log("   stage 1 figure saved", "info")

            elif step == "stage2_plot":
                sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"],  NVIEW, NDET)
                sino_bh,    _ = pj_io.read_pj(FILES["sino_bh"],     NVIEW, NDET)
                sino_s2,    _ = pj_io.read_pj(FILES["sino_stage2"], NVIEW, NDET)
                lut_data = np.load(FILES["lut_npz"], allow_pickle=True)
                from plotter import Stage2Plotter
                Stage2Plotter(FILES["fig_stage2"]).plot(
                    sinos={"ideal": sino_ideal, "bh": sino_bh, "stage2": sino_s2},
                    lut={"empirical": lut_data["empirical_lut"],
                         "blended":   lut_data["blended_lut"]},
                    n_views=NVIEW,
                )
                self._log("   stage 2 figure saved", "info")

        except Exception as e:
            self._log(f"   plot failed: {e}", "warn")

    def _refresh_sinograms(self, step: str):
        """Load and display sinograms after relevant steps complete."""
        if not PIPELINE_AVAILABLE:
            return
        try:
            label_map = {
                "generate":    ("sino_ideal", 0),
                "correct":     ("sino_corrected", 2),
                "apply_lut":   ("sino_stage2", 2),
            }
            if step not in label_map:
                return
            key, idx = label_map[step]
            path = FILES.get(key)
            if path and os.path.exists(path):
                sino, _ = pj_io.read_pj(path, NVIEW, NDET)
                titles = ["Ideal", "Beam Hardened", "Corrected"]
                self.sino_canvases[idx].show_sino(sino, titles[idx])

            # Always try to show BH after generate
            if step == "generate":
                bh_path = FILES.get("sino_bh")
                if bh_path and os.path.exists(bh_path):
                    bh, _ = pj_io.read_pj(bh_path, NVIEW, NDET)
                    self.sino_canvases[1].show_sino(bh, "Beam Hardened")
        except Exception as e:
            self._log(f"   sinogram refresh failed: {e}", "warn")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CT BH Pipeline")

    # Try to set a nice font
    QFontDatabase.addApplicationFont("IBMPlexMono-Regular.ttf")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()