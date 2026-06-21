from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame,
)
from PySide6.QtCore import Qt, Signal

from interface.styles import COLORS

class StartPanel(QWidget):

    analyse_requested=Signal()
    compare_requested=Signal()

    def __init__(self, parent=None)->None:
        super().__init__(parent)
        self.buildUi()
    
    def buildUi(self) -> None:
        base=QVBoxLayout(self)
        base.setContentsMargins(60, 60, 60, 60)
        base.setSpacing(0)
        base.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title=QLabel("Domain Analysis Tool")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {COLORS['text']};")

        subtitle=QLabel("Choose what you want to do")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"font-size: 14px; color: {COLORS['text_secondary']};")

        options=QVBoxLayout()
        options.setSpacing(16)
        options.addWidget(self.buildOptionRow(
            "Analyse a single image",
            "Load or create a segmentation session to explore the domains of a single image in detail.",
            "Open",
            self.analyse_requested,
        ))
        options.addWidget(self.buildOptionRow(
            "Compare domains",
            "Load two sessions and compare the statistics and displacement of selected domains side by side.",
            "Compare",
            self.compare_requested,
        ))

        base.addWidget(title)
        base.addWidget(subtitle)
        base.addSpacing(32)
        base.addLayout(options)
    
    def buildOptionRow(self, title:str, description:str, btn_text:str, signal:Signal)->QFrame:
        frame=QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 10px; }}"
        )
        row=QHBoxLayout(frame)
        row.setContentsMargins(20, 20, 20, 20)
        row.setSpacing(24)

        text_col=QVBoxLayout()
        text_col.setSpacing(6)

        title_lbl=QLabel(title)
        title_lbl.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {COLORS['text']}; border: none;")

        desc_lbl=QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(f"font-size: 13px; color: {COLORS['text_secondary']}; border: none;")

        text_col.addWidget(title_lbl)
        text_col.addWidget(desc_lbl)

        but=QPushButton(btn_text)
        but.setFixedWidth(120)
        but.clicked.connect(signal)

        row.addLayout(text_col, stretch=1)
        row.addWidget(but)
        return frame
    
    