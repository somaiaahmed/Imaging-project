from __future__ import annotations

import os
import sys
import time
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QStackedWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import pj_io
    from main import (
        run_generate,
        run_calibrate,
        run_correct,
        run_build_lut,
        run_apply_lut,
        run_bone_correct,
        FILES,
        NVIEW,
        NDET,
    )
    from reconstruction import FBPReconstructor
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    NVIEW, NDET = 360, 512
    FILES = {}


DARK = {
    "bg": "#0d1117",
    "bg2": "#111827",
    "bg3": "#172033",
    "bg4": "#1e293b",
    "border": "#263244",
    "border2": "#334155",
    "text": "#e5e7eb",
    "text2": "#94a3b8",
    "text3": "#64748b",
    "accent": "#3b82f6",
    "accent2": "#2563eb",
    "accent_dim": "#172554",
    "green": "#22c55e",
    "green_dim": "#052e16",
    "amber": "#f59e0b",
    "amber_dim": "#451a03",
    "red": "#ef4444",
    "red_dim": "#450a0a",
    "cyan": "#22d3ee",
    "cyan_dim": "#083344",
}


STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK['bg']};
    color: {DARK['text']};
    font-family: 'IBM Plex Mono', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QWidget#Sidebar {{
    background-color: {DARK['bg2']};
    border-right: 1px solid {DARK['border']};
}}
QWidget#LeftPanel {{
    background-color: {DARK['bg2']};
    border-right: 1px solid {DARK['border']};
}}
QWidget#RightPanel {{
    background-color: {DARK['bg']};
}}
QPushButton {{
    background-color: {DARK['bg3']};
    color: {DARK['text']};
    border: 1px solid {DARK['border2']};
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {DARK['bg4']};
    border-color: {DARK['accent']};
}}
QPushButton:pressed {{
    background-color: {DARK['accent_dim']};
}}
QPushButton#PrimaryBtn {{
    background-color: {DARK['accent']};
    border: none;
    color: white;
    font-weight: bold;
}}
QPushButton#PrimaryBtn:hover {{
    background-color: {DARK['accent2']};
}}
QPushButton:checked {{
    background-color: {DARK['accent']};
    border-color: {DARK['accent2']};
    color: white;
    font-weight: bold;
}}
QTextEdit#Console {{
    background-color: {DARK['bg']};
    color: {DARK['text2']};
    border: none;
    border-top: 1px solid {DARK['border']};
    font-family: 'IBM Plex Mono', 'Consolas', monospace;
    font-size: 11px;
    padding: 6px 10px;
}}
QScrollBar:vertical {{
    background: {DARK['bg2']};
    width: 7px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {DARK['border2']};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QProgressBar {{
    background: {DARK['border']};
    border: none;
    border-radius: 2px;
    height: 4px;
}}
QProgressBar::chunk {{
    background: {DARK['accent']};
    border-radius: 2px;
}}
"""


class PipelineWorker(QThread):
    log_signal = pyqtSignal(str, str)
    done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str, str)

    def __init__(self, steps: list[str]):
        super().__init__()
        self.steps = steps

    def _call(self, step: str):
        if not PIPELINE_AVAILABLE:
            time.sleep(0.5)
            return

        if step == "generate":
            run_generate()
        elif step == "calibrate":
            run_calibrate()
        elif step == "correct":
            run_correct(show_plot=False)
        elif step == "build_lut":
            run_build_lut()
        elif step == "apply_lut":
            run_apply_lut()
        elif step == "bone_correct":
            run_bone_correct()
        else:
            raise ValueError(f"Unknown step: {step}")

    def run(self):
        for step in self.steps:
            t0 = time.time()
            self.log_signal.emit(f"running {step} ...", "accent")
            try:
                self._call(step)
                elapsed = time.time() - t0
                self.log_signal.emit(f"{step} done in {elapsed:.2f}s", "ok")
                self.done_signal.emit(step)
            except Exception as exc:
                self.log_signal.emit(f"{step} failed: {exc}", "error")
                self.error_signal.emit(step, str(exc))
                return


class SinogramCanvas(FigureCanvas):
    def __init__(self, title: str = "", parent=None):
        self.fig = Figure(figsize=(2.8, 3.2), dpi=96, facecolor=DARK["bg"])
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._style_ax(title)

    def _style_ax(self, title: str):
        self.ax.set_facecolor(DARK["bg"])
        for spine in self.ax.spines.values():
            spine.set_edgecolor(DARK["border"])
        self.ax.tick_params(colors=DARK["text3"], labelsize=7)
        self.ax.set_title(title, color=DARK["text2"], fontsize=9, pad=6)

    def show_placeholder(self, title: str):
        self.ax.clear()
        self._style_ax(title)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(
            0.5,
            0.5,
            "no result",
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            color=DARK["text3"],
            fontsize=11,
        )
        self.draw()

    def show_sino(self, sino: np.ndarray, title: str):
        self.ax.clear()
        self._style_ax(title)
        lo = float(np.percentile(sino, 1))
        hi = float(np.percentile(sino, 99))
        self.ax.imshow(sino, cmap="inferno", aspect="auto", vmin=lo, vmax=hi, origin="upper")
        self.ax.set_xlabel("Detector", color=DARK["text3"], fontsize=7)
        self.ax.set_ylabel("View", color=DARK["text3"], fontsize=7)
        self.draw()

    def show_image(self, image: np.ndarray, title: str):
        self.ax.clear()
        self._style_ax(title)
        lo = float(np.percentile(image, 1))
        hi = float(np.percentile(image, 99))
        self.ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, origin="upper")
        self.ax.set_xlabel("X", color=DARK["text3"], fontsize=7)
        self.ax.set_ylabel("Y", color=DARK["text3"], fontsize=7)
        self.draw()


class MainWindow(QMainWindow):
    LOG_COLORS = {
        "ok": DARK["green"],
        "warn": DARK["amber"],
        "info": DARK["text2"],
        "accent": DARK["accent"],
        "error": DARK["red"],
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Beam Hardening Pipeline")
        self.resize(1380, 820)
        self.setStyleSheet(STYLESHEET)

        self._worker: PipelineWorker | None = None
        self._display_mode = "all"

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
        splitter.setSizes([320, 1130])
        root.addWidget(splitter, 1)

    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Sidebar")
        w.setFixedWidth(52)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        def icon_btn(symbol: str, active: bool = False):
            btn = QToolButton()
            btn.setText(symbol)
            btn.setFixedSize(52, 52)
            color = DARK["accent"] if active else DARK["text3"]
            btn.setStyleSheet(
                f"QToolButton {{ background: transparent; color: {color}; "
                f"font-size: 16px; border: none; "
                f"border-left: 2px solid {DARK['accent'] if active else 'transparent'}; }}"
                f"QToolButton:hover {{ background: {DARK['bg3']}; color: {DARK['text']}; }}"
            )
            return btn

        logo = QLabel("[]")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(52, 52)
        logo.setStyleSheet(f"color: {DARK['accent']}; font-size: 18px; font-weight: bold;")
        lay.addWidget(logo)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {DARK['border']};")
        lay.addWidget(sep)
        lay.addSpacing(8)
        lay.addWidget(icon_btn(">", active=True))
        lay.addWidget(icon_btn("[]"))
        lay.addWidget(icon_btn("~"))
        lay.addStretch()
        return w

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("LeftPanel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(48)
        hdr.setStyleSheet(f"background: {DARK['bg2']}; border-bottom: 1px solid {DARK['border']};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(14, 0, 14, 0)
        title = QLabel("PIPELINE")
        title.setStyleSheet(f"color: {DARK['text3']}; font-size: 10px; font-weight: bold; letter-spacing: 2px;")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()
        lay.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(14, 14, 14, 14)
        inner_lay.setSpacing(12)

        intro = QLabel(
            "Please Choose a Run Mode"
        )
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignCenter)
        intro.setStyleSheet(
            f"background: {DARK['bg3']}; border: 1px solid {DARK['border']}; "
            f"border-radius: 10px; padding: 12px; color: {DARK['text2']}; font-size: 11px;"
        )
        inner_lay.addWidget(intro)

        self.run_stage1_btn = QPushButton("Run Stage 1")
        self.run_stage1_btn.setFixedHeight(40)
        self.run_stage1_btn.clicked.connect(self._run_stage1)

        self.run_stage2_btn = QPushButton("Run Stage 2")
        self.run_stage2_btn.setFixedHeight(40)
        self.run_stage2_btn.clicked.connect(self._run_stage2)

        self.run_stage3_btn = QPushButton("Run Stage 3")
        self.run_stage3_btn.setFixedHeight(40)
        self.run_stage3_btn.clicked.connect(self._run_stage3)

        self.run_all_btn = QPushButton("Run All Stages")
        self.run_all_btn.setFixedHeight(40)
        self.run_all_btn.clicked.connect(self._run_all)

        self.stage_buttons = [
            self.run_stage1_btn,
            self.run_stage2_btn,
            self.run_stage3_btn,
            self.run_all_btn,
        ]
        for btn in self.stage_buttons:
            btn.setCheckable(True)
            inner_lay.addWidget(btn)

        inner_lay.addStretch()

        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(4)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("RightPanel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(self._build_topbar())

        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {DARK['border']}; }}")
        splitter.addWidget(self._build_main_views())
        splitter.addWidget(self._build_console())
        splitter.setSizes([620, 180])
        lay.addWidget(splitter, 1)
        return w

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setStyleSheet(f"background: {DARK['bg2']}; border-bottom: 1px solid {DARK['border']};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        title = QLabel("Beam Hardening Results")
        title.setStyleSheet(f"color: {DARK['text']}; font-size: 14px; font-weight: bold;")
        subtitle = QLabel("Ideal, input, and stage outputs")
        subtitle.setStyleSheet(f"color: {DARK['text3']}; font-size: 11px;")
        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addStretch()

        self.results_view_btn = QPushButton("Results")
        self.results_view_btn.setCheckable(True)
        self.results_view_btn.setChecked(True)
        self.results_view_btn.clicked.connect(lambda: self._set_main_view("results"))
        lay.addWidget(self.results_view_btn)

        self.recon_view_btn = QPushButton("Reconstructions")
        self.recon_view_btn.setCheckable(True)
        self.recon_view_btn.clicked.connect(lambda: self._set_main_view("recon"))
        lay.addWidget(self.recon_view_btn)

        clear_btn = QPushButton("Clear Log")
        clear_btn.setFixedHeight(30)
        clear_btn.clicked.connect(self._clear_log)
        lay.addWidget(clear_btn)
        return bar

    def _build_results_board(self) -> QWidget:
        board = QWidget()
        lay = QVBoxLayout(board)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        heading = QLabel("Results Overview")
        heading.setStyleSheet(f"color: {DARK['text']}; font-size: 16px; font-weight: bold;")

        lay.addWidget(heading)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        self.result_canvases: dict[str, SinogramCanvas] = {}
        cards = [
            ("ideal", "Ideal", "Ground truth sinogram", DARK["green"], 0),
            ("bh", "Beam Hardening", "Corrupted input projection", DARK["red"], 1),
            ("stage1", "Stage 1", "Polynomial correction output", DARK["accent"], 2),
            ("stage2", "Stage 2", "Empirical LUT correction", DARK["amber"], 3),
            ("stage3", "Stage 3", "Bone Correction", DARK["cyan"], 4),
        ]

        for key, title, note, color, col in cards:
            card, canvas = self._build_result_card(title, note, color)
            self.result_canvases[key] = canvas
            grid.addWidget(card, 0, col)

        for idx in range(5):
            grid.setColumnStretch(idx, 1)
        grid.setRowStretch(0, 1)

        lay.addLayout(grid, 1)
        self._clear_result_canvases()
        return board

    def _build_recon_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        heading = QLabel("Reconstructions")
        heading.setStyleSheet(f"color: {DARK['text']}; font-size: 16px; font-weight: bold;")
        lay.addWidget(heading)

        wrap = QWidget()
        wrap.setStyleSheet(
            f"background: {DARK['bg2']}; border: 1px solid {DARK['border']}; border-radius: 10px;"
        )
        wrap_lay = QVBoxLayout(wrap)
        wrap_lay.setContentsMargins(0, 0, 0, 0)
        wrap_lay.setSpacing(0)

        hdr = QLabel("RECONSTRUCTIONS")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(
            f"background: {DARK['bg3']}; color: {DARK['text']}; font-size: 10px; "
            f"font-weight: bold; letter-spacing: 2px; border-bottom: 1px solid {DARK['border']};"
        )
        wrap_lay.addWidget(hdr)

        sub = QLabel("FBP preview for each available result")
        sub.setAlignment(Qt.AlignCenter)
        sub.setWordWrap(True)
        sub.setFixedHeight(36)
        sub.setStyleSheet(
            f"background: {DARK['bg2']}; color: {DARK['text3']}; font-size: 9px; "
            f"padding: 6px 8px; border-bottom: 1px solid {DARK['border']};"
        )
        wrap_lay.addWidget(sub)

        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.recon_canvases: dict[str, SinogramCanvas] = {}
        recon_cards = [
            ("ideal", "Ideal"),
            ("bh", "Beam Hardening"),
            ("stage1", "Stage 1"),
            ("stage2", "Stage 2"),
            ("stage3", "Stage 3"),
        ]
        # for idx, (key, title) in enumerate(recon_cards):
        #     card, canvas = self._build_result_card(title, "FBP reconstruction", DARK["text2"])
        #     self.recon_canvases[key] = canvas
        #     grid.addWidget(card, idx // 3, idx % 3)
        positions = {
            "ideal": (0, 0),
            "bh": (0, 1),

            "stage1": (1, 0),
            "stage2": (1, 1),
            "stage3": (1, 2),
    }

        for key, title in recon_cards:
            card, canvas = self._build_result_card(title, "FBP reconstruction", DARK["text2"])
            self.recon_canvases[key] = canvas

            row, col = positions[key]
            grid.addWidget(card, row, col)    
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        wrap_lay.addWidget(inner, 1)
        lay.addWidget(wrap, 1)
        self._clear_recon_canvases()
        return panel

    def _build_main_views(self) -> QWidget:
        self.main_views = QStackedWidget()
        self.results_board = self._build_results_board()
        self.recon_board = self._build_recon_panel()
        self.main_views.addWidget(self.results_board)
        self.main_views.addWidget(self.recon_board)
        return self.main_views

    def _set_main_view(self, view: str):
        is_results = view == "results"
        self.results_view_btn.setChecked(is_results)
        self.recon_view_btn.setChecked(not is_results)
        self.main_views.setCurrentWidget(self.results_board if is_results else self.recon_board)

    def _build_result_card(self, title: str, note: str, color: str):
        cell = QWidget()
        cell.setMinimumHeight(300)
        cell.setStyleSheet(
            f"background: {DARK['bg2']}; border: 1px solid {DARK['border']}; border-radius: 10px;"
        )
        cl = QVBoxLayout(cell)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        hdr = QLabel(title.upper())
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setFixedHeight(34)
        hdr.setStyleSheet(
            f"background: {DARK['bg3']}; color: {color}; font-size: 10px; "
            f"font-weight: bold; letter-spacing: 2px; border-bottom: 1px solid {DARK['border']};"
        )
        note_lbl = QLabel(note)
        note_lbl.setAlignment(Qt.AlignCenter)
        note_lbl.setWordWrap(True)
        note_lbl.setFixedHeight(36)
        note_lbl.setStyleSheet(
            f"background: {DARK['bg2']}; color: {DARK['text3']}; font-size: 9px; "
            f"padding: 6px 10px; border-bottom: 1px solid {DARK['border']};"
        )
        canvas = SinogramCanvas(title, parent=cell)
        cl.addWidget(hdr)
        cl.addWidget(note_lbl)
        cl.addWidget(canvas, 1)
        return cell, canvas

    def _build_console(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(30)
        hdr.setStyleSheet(
            f"background: {DARK['bg2']}; border-top: 1px solid {DARK['border']}; "
            f"border-bottom: 1px solid {DARK['border']};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 12, 0)
        lbl = QLabel("CONSOLE")
        lbl.setStyleSheet(f"color: {DARK['text3']}; font-size: 10px; font-weight: bold; letter-spacing: 2px;")
        hl.addWidget(lbl)
        hl.addStretch()
        lay.addWidget(hdr)

        self.console = QTextEdit()
        self.console.setObjectName("Console")
        self.console.setReadOnly(True)
        lay.addWidget(self.console, 1)

        self._log("ready. choose a stage to run.", "info")
        return w

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

    def _set_buttons_enabled(self, enabled: bool):
        for btn in self.stage_buttons:
            btn.setEnabled(enabled)
        self.progress.setVisible(not enabled)

    def _set_active_button(self, active_btn: QPushButton):
        for btn in self.stage_buttons:
            btn.setChecked(btn is active_btn)

    def _clear_result_canvases(self):
        titles = {
            "ideal": "Ideal",
            "bh": "Beam Hardening",
            "stage1": "Stage 1",
            "stage2": "Stage 2",
            "stage3": "Stage 3",
        }
        for key, canvas in self.result_canvases.items():
            canvas.show_placeholder(titles[key])

    def _clear_recon_canvases(self):
        titles = {
            "ideal": "Ideal",
            "bh": "Beam Hardening",
            "stage1": "Stage 1",
            "stage2": "Stage 2",
            "stage3": "Stage 3",
        }
        for key, canvas in self.recon_canvases.items():
            canvas.show_placeholder(titles[key])

    def _prepare_display_mode(self, mode: str):
        self._display_mode = mode
        self._clear_result_canvases()
        self._clear_recon_canvases()
        self._log(f"display mode: {mode}", "info")

    def _run_steps(self, steps: list[str]):
        if self._worker and self._worker.isRunning():
            return

        self._set_buttons_enabled(False)
        self._log(f"running steps: {', '.join(steps)}", "info")
        self._worker = PipelineWorker(steps)
        self._worker.log_signal.connect(self._log)
        self._worker.done_signal.connect(self._on_step_done)
        self._worker.error_signal.connect(self._on_step_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _run_stage1(self):
        self._set_active_button(self.run_stage1_btn)
        self._prepare_display_mode("stage1")
        self._run_steps(["generate", "calibrate", "correct"])

    def _run_stage2(self):
        self._set_active_button(self.run_stage2_btn)
        self._prepare_display_mode("stage2")
        self._run_steps(["generate", "build_lut", "apply_lut"])

    def _run_stage3(self):
        self._set_active_button(self.run_stage3_btn)
        self._prepare_display_mode("stage3")
        self._run_steps(["generate", "bone_correct"])

    def _run_all(self):
        self._set_active_button(self.run_all_btn)
        self._prepare_display_mode("all")
        self._run_steps(["generate", "calibrate", "correct", "build_lut", "apply_lut", "bone_correct"])

    def _on_step_done(self, step: str):
        self._refresh_results(step)
        self._refresh_reconstructions(step)

    def _on_step_error(self, step: str, err: str):
        self._log(f"stopped after {step}: {err}", "error")

    def _on_worker_finished(self):
        self._set_buttons_enabled(True)
        self._log("run complete.", "ok")

    def _refresh_results(self, step: str):
        if not PIPELINE_AVAILABLE:
            return

        try:
            if step == "generate":
                for key, file_key, title in [
                    ("ideal", "sino_ideal", "Ideal"),
                    ("bh", "sino_bh", "Beam Hardening"),
                ]:
                    path = FILES.get(file_key)
                    if path and os.path.exists(path):
                        sino, _ = pj_io.read_pj(path, NVIEW, NDET)
                        self.result_canvases[key].show_sino(sino, title)

            elif step == "correct" and self._display_mode in {"stage1", "all"}:
                path = FILES.get("sino_corrected")
                if path and os.path.exists(path):
                    sino, _ = pj_io.read_pj(path, NVIEW, NDET)
                    self.result_canvases["stage1"].show_sino(sino, "Stage 1")

            elif step == "apply_lut":
                if self._display_mode in {"stage2", "all"}:
                    path = FILES.get("sino_stage2")
                    if path and os.path.exists(path):
                        sino, _ = pj_io.read_pj(path, NVIEW, NDET)
                        self.result_canvases["stage2"].show_sino(sino, "Stage 2")

            elif step == "bone_correct" and self._display_mode in {"stage3", "all"}:
                    path = FILES.get("sino_bone")
                
                    if path and os.path.exists(path):
                        print("found the path")
                        sino, _ = pj_io.read_pj(path, NVIEW, NDET)
                        self.result_canvases["stage3"].show_sino(sino, "Stage 3")
        except Exception as exc:
            self._log(f"result refresh failed: {exc}", "warn")

    def _refresh_reconstructions(self, step: str):
        if not PIPELINE_AVAILABLE or step not in {"generate", "correct", "apply_lut","bone_correct"}:
            return

        try:
            sinograms: dict[str, np.ndarray] = {}

            ideal_path = FILES.get("sino_ideal")
            bh_path = FILES.get("sino_bh")
            if ideal_path and os.path.exists(ideal_path):
                sinograms["ideal"], _ = pj_io.read_pj(ideal_path, NVIEW, NDET)
            if bh_path and os.path.exists(bh_path):
                sinograms["bh"], _ = pj_io.read_pj(bh_path, NVIEW, NDET)

            if self._display_mode in {"stage1", "all"}:
                path = FILES.get("sino_corrected")
                if path and os.path.exists(path):
                    sinograms["stage1"], _ = pj_io.read_pj(path, NVIEW, NDET)

            if self._display_mode in {"stage2", "all"}:
                path = FILES.get("sino_stage2")
                if path and os.path.exists(path):
                    sinograms["stage2"], _ = pj_io.read_pj(path, NVIEW, NDET)

            if self._display_mode in {"stage3", "all"}:
                path = FILES.get("sino_bone")
                if path and os.path.exists(path):
                    sinograms["stage3"], _ = pj_io.read_pj(path, NVIEW, NDET)

            if not sinograms:
                return

            recon = FBPReconstructor().reconstruct_many(**sinograms)
            for key, image in recon.items():
                if key in self.recon_canvases:
                    title = {
                        "ideal": "Ideal",
                        "bh": "Beam Hardening",
                        "stage1": "Stage 1",
                        "stage2": "Stage 2",
                        "stage3": "Stage 3",
                    }[key]
                    self.recon_canvases[key].show_image(image, title)
        except Exception as exc:
            self._log(f"reconstruction refresh failed: {exc}", "warn")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CT BH Pipeline")
    QFontDatabase.addApplicationFont("IBMPlexMono-Regular.ttf")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()