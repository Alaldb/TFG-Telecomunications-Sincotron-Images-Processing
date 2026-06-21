from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QSizePolicy, QFrame, QMessageBox,
    QRadioButton, QButtonGroup
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QSize, Qt, Signal
from core.session import Session
from interface.styles import COLORS
from persistence.session_io import loadSession


class LoadPanel(QWidget):
    image_loaded = Signal(np.ndarray, str) #Create signal that allows the creation of the widget in the main window with the image and its name
    session_loaded = Signal(Session) #Create signal that allows the creation of the widget in the main window with the session data when loading a session
    home = Signal()

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

        self.btn_load_image = QPushButton("Load image")
        self.btn_load_image.setFixedWidth(180)
        self.btn_load_image.setFixedHeight(42)
        self.btn_load_image.clicked.connect(self.onLoadImageClicked)

        self.btn_load_session = QPushButton("Load session")
        self.btn_load_session.setFixedWidth(180)
        self.btn_load_session.setFixedHeight(42)
        self.btn_load_session.clicked.connect(self.onLoadSessionClicked)

        option_row = QHBoxLayout()
        option_row.addStretch()
        option_row.addWidget(self.btn_load_image)
        option_row.addSpacing(24)
        option_row.addWidget(self.btn_load_session)
        option_row.addStretch()

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()

        self.home_but = QPushButton("Home")
        self.home_but.setFixedWidth(120)
        self.home_but.setObjectName("cancel_btn")
        self.home_but.clicked.connect(self.home)
        bottom_row.addWidget(self.home_but)
        bottom_row.addStretch()

        empty_layout.addWidget(title)
        empty_layout.addWidget(subtitle)
        empty_layout.addSpacing(24)
        empty_layout.addLayout(option_row)
        empty_layout.addLayout(bottom_row)

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
        self.image_view.setMinimumHeight(300)
        frame_layout.addWidget(self.image_view)

        action_row=QHBoxLayout()
        action_row.addStretch()

        self.btn_load_image2 = QPushButton("Load image")
        self.btn_load_image2.setFixedWidth(180)
        self.btn_load_image2.setFixedHeight(42)
        self.btn_load_image2.clicked.connect(self.onLoadImageClicked)

        self.btn_load_session2 = QPushButton("Load session")
        self.btn_load_session2.setFixedWidth(180)
        self.btn_load_session2.setFixedHeight(42)
        self.btn_load_session2.clicked.connect(self.onLoadSessionClicked)

        action_row.addWidget(self.btn_load_image2)
        action_row.addSpacing(8)
        action_row.addWidget(self.btn_load_session2)
        action_row.addStretch()

        method_row=QHBoxLayout()
        method_row.addStretch()

        self.method_group=QButtonGroup(self)

        self.btn_icm=QRadioButton("ICM")
        self.btn_icm.setChecked(True)

        self.btn_gc=QRadioButton("Graph Cuts")

        self.method_group.addButton(self.btn_icm)
        self.method_group.addButton(self.btn_gc)

        method_row.addWidget(self.btn_icm)
        method_row.addSpacing(16)
        method_row.addWidget(self.btn_gc)
        method_row.addStretch()

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.next_but = QPushButton("Next ->")
        self.next_but.setFixedWidth(120)
        

        self.home_but = QPushButton("Home")
        self.home_but.setFixedWidth(120)
        self.home_but.setObjectName("cancel_btn")
        self.home_but.clicked.connect(self.home)
        bottom_row.addWidget(self.home_but)
        bottom_row.addWidget(self.next_but)
        bottom_row.addStretch()

        loaded_layout.addWidget(self.file_name)
        loaded_layout.addWidget(image_frame, stretch=1)
        loaded_layout.addLayout(action_row)
        loaded_layout.addLayout(method_row)
        loaded_layout.addLayout(bottom_row)
        

        main_layout.addWidget(self.loaded_state)

    def onLoadImageClicked(self):
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

        name_with_ext = Path(path).name
        name = name_with_ext.rsplit(".", 1)[0]

        self.empty_state.hide()
        self.loaded_state.show()

        self.file_name.setText(name)
        self.showImage(image)
        self.image_loaded.emit(image, name)
    
    def onLoadSessionClicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select session", "",
            "Session files (*.session)"
        )
        if not path:
            return
        try:
            session=loadSession(path)
            self.session_loaded.emit(session)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error loading session",
                f"Could not load the session:\n{path}\n\nThe file may be corrupted or in an unsupported format.\n\nError details:\n{str(e)}"
            )


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
    
    @property
    def selected_method(self) -> str:
        return "ICM" if self.btn_icm.isChecked() else "GraphCuts"
