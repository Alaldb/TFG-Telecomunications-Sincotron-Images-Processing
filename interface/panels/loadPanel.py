from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QSizePolicy, QFrame, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QSize, Qt, Signal
from interface.styles import COLORS


class LoadPanel(QWidget):
    image_loaded = Signal(np.ndarray, str) #Create signal that allows the creation of the widget in the main window with the image and its name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.buildUi()

    def buildUi(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(60, 60, 60, 60)
        main_layout.setSpacing(0)

        # Empty State
        self.empty_state = QWidget()
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setSpacing(16)
        empty_layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Domain Analysis Tool")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 26px; font-weight: bold; color: {COLORS['text']};"
        )

        subtitle = QLabel("Load an image to begin analysis")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: 14px; color: {COLORS['text_secondary']};"
        )

        self.btn_load = QPushButton("Load image")
        self.btn_load.setFixedWidth(180)
        self.btn_load.setFixedHeight(42)
        self.btn_load.clicked.connect(self.onLoadClicked)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.btn_load)
        btn_row.addStretch()

        empty_layout.addWidget(title)
        empty_layout.addWidget(subtitle)
        empty_layout.addSpacing(24)
        empty_layout.addLayout(btn_row)

        main_layout.addStretch()
        main_layout.addWidget(self.empty_state)
        main_layout.addStretch()

        # Image load
        self.loaded_state = QWidget()
        self.loaded_state.hide()
        loaded_layout = QVBoxLayout(self.loaded_state)
        loaded_layout.setContentsMargins(0, 0, 0, 0)
        loaded_layout.setSpacing(12)

        self.file_name = QLabel("")
        self.file_name.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {COLORS['text']};"
        )

        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_frame.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                background-color: {COLORS['panel']};
            }}
        """)
        frame_layout = QVBoxLayout(image_frame)
        frame_layout.setContentsMargins(12, 12, 12, 12)

        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_view.setMinimumHeight(500)
        frame_layout.addWidget(self.image_view)

        self.btn_load2 = QPushButton("Load image")
        self.btn_load2.setFixedWidth(180)
        self.btn_load2.setFixedHeight(42)
        self.btn_load2.clicked.connect(self.onLoadClicked)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.next_but = QPushButton("Next →")
        self.next_but.setFixedWidth(120)
        bottom_row.addWidget(self.next_but)

        loaded_layout.addWidget(self.file_name)
        loaded_layout.addWidget(image_frame, stretch=1)
        loaded_layout.addWidget(self.btn_load2)
        loaded_layout.addLayout(bottom_row)

        main_layout.addWidget(self.loaded_state)

    def onLoadClicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Image files (*.tif *.tiff)"
        )
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            QMessageBox.critical(
                self,
                "Error loading image",
                f"Could not read the file:\n{path}\n\nThe file may be corrupted or in an unsupported format."
            )
            return

        name_with_ext = path.split("/")[-1].split("\\")[-1]
        name = name_with_ext.rsplit(".", 1)[0]

        self.empty_state.hide()
        self.loaded_state.show()

        self.file_name.setText(name)
        self.showImage(image)
        self.image_loaded.emit(image, name)

    def showImage(self, image: np.ndarray):
        image_8 = self.toUint8(image)
        height, width = image_8.shape
        q_image = QImage(image_8.data, width, height, width,
                         QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        pixmap = QPixmap.fromImage(q_image).scaled(
            QSize(self.image_view.width() or 550, self.image_view.height() or 350),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_view.setPixmap(pixmap)

    def toUint8(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        return img.astype(np.uint8)