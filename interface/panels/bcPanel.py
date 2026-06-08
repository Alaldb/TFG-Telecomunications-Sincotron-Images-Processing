
from __future__ import annotations
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QLineEdit,
    QSizePolicy, QFrame,QCheckBox,QScrollArea
)
from PySide6.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator
from PySide6.QtCore import QLocale, QSize, Qt, Signal
from interface.styles import COLORS
from processing.corrector import Corrector

class BcPanel(QWidget):
    correction_accepted=Signal(np.ndarray,int,int)
    cancelled=Signal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.corrector=Corrector()
        self.image: np.ndarray|None=None
        self.corrected: np.ndarray|None=None
        self.v_low: int=0
        self.v_high: int=255
        self.brightness: int=0
        self.contrast: float=1.0
        self.updating=False
        self.crop_enabled: bool=False
        self.crop_px: int=5
        self.buildUi()
    
    #Functionality of the Panel, see Corrector class

    def defaultCorrection(self, image: np.ndarray) -> None:
        self.image=self.corrector.normalize_to_uint8(image)
        self.v_low,self.v_high=self.corrector.histogram_range(self.image, self.corrector.coverage)
        self.updateValues()
        self.applyCorrection()
        self.updateHistogram()

    def buildUi(self) -> None:
        base_layout=QVBoxLayout(self)
        base_layout.setContentsMargins(24,24,24,24)
        base_layout.setSpacing(16)

        content=QHBoxLayout()
        content.setSpacing(20)
        content.addWidget(self.buildLeftPanel(),stretch=3)
        content.addWidget(self.buildRightPanel(),stretch=0)
        base_layout.addLayout(content, stretch=1)
        base_layout.addLayout(self.buildBottomRow())

    def buildLeftPanel(self) -> QWidget:
        container=QWidget()
        left_layout=QVBoxLayout(container)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(12)
        #Frame Component for image
        image_frame=QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_frame.setStyleSheet(f"""
            QFrame{{
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                background-color: {COLORS['panel']};
            }}
            """)
        frame_layout=QVBoxLayout(image_frame)
        frame_layout.setContentsMargins(12,12,12,12)
        #Image
        self.image_view=QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setMinimumHeight(300)
        frame_layout.addWidget(self.image_view)
        #Histogram
        self.figure=Figure(figsize=(5,2), tight_layout=True)
        self.figure.patch.set_facecolor(COLORS['panel'])#change plot background
        self.ax=self.figure.add_subplot(111)#1 plot in the figure
        self.canvas=FigureCanvasQTAgg(self.figure)#we translate the class from matplotlib to Qt (Pyside6) as a widget
        self.canvas.setMinimumHeight(160)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        left_layout.addWidget(image_frame, stretch=3)
        left_layout.addWidget(self.canvas, stretch=0)

        return container

    def buildRightPanel(self) -> QWidget:
        container=QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setFixedWidth(260)
        container.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {COLORS['border']};
                background-color: {COLORS['panel']};               
            }}
        """)
        right_layout=QVBoxLayout(container)
        right_layout.setContentsMargins(16,20,16,20)
        right_layout.setSpacing(12)

        title=QLabel("Brightness and Contrast")
        title.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {COLORS['text']}; border: none;"
        )

        #Low intensity end
        vlow_label=QLabel("Low intensity end value")
        vlow_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text']}; border: none;")
        self.input_vlow=QLineEdit(f"{self.v_low}")
        self.input_vlow.setFixedWidth(60)
        self.input_vlow.setValidator(QIntValidator(0,255))#Always between 0 an 255 with int values
        self.vlow_slider=QSlider(Qt.Horizontal)
        self.vlow_slider.setRange(0,255)
        vlow_row=QHBoxLayout()
        vlow_row.addWidget(vlow_label)
        vlow_row.addStretch()
        vlow_row.addWidget(self.input_vlow)

        #High intensity end
        vhigh_label=QLabel("High intensity end value")
        vhigh_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text']}; border: none;")
        self.input_vhigh=QLineEdit(f"{self.v_high}")
        self.input_vhigh.setFixedWidth(60)
        self.input_vhigh.setValidator(QIntValidator(0,255))#Always between 0 an 255 with int values
        self.vhigh_slider=QSlider(Qt.Horizontal)
        self.vhigh_slider.setRange(0,255)
        vhigh_row=QHBoxLayout()
        vhigh_row.addWidget(vhigh_label)
        vhigh_row.addStretch()
        vhigh_row.addWidget(self.input_vhigh)

        #Brightness input
        brightness_label=QLabel("Brightness value")
        brightness_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text']}; border: none;")
        self.input_brightness=QLineEdit(f"{self.brightness}")
        self.input_brightness.setFixedWidth(60)
        self.input_brightness.setValidator(QIntValidator(-127,127))
        brightness_row=QHBoxLayout()
        brightness_row.addWidget(brightness_label)
        brightness_row.addStretch()
        brightness_row.addWidget(self.input_brightness)

        #Contrast input
        contrast_label=QLabel("Contrast value")
        contrast_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text']}; border: none;")
        self.input_contrast=QLineEdit(f"{self.contrast}")
        self.input_contrast.setFixedWidth(60)
        #self.input_contrast.setValidator(QDoubleValidator(0.1, 10.0, 2))
        contrast_row=QHBoxLayout()
        contrast_row.addWidget(contrast_label)
        contrast_row.addStretch()
        contrast_row.addWidget(self.input_contrast)

        #Reset Button
        self.reset_but=QPushButton("Reset to default values")
        self.reset_but.clicked.connect(self.onReset)

        separator2=QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setStyleSheet(f"color: {COLORS['border']}; border: none; border-top: 1px solid {COLORS['border']};")

        #Crop Input
        crop_label=QLabel("Crop borders")
        crop_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text']}; border: none;")
        self.crop_check=QCheckBox()
        self.crop_check.setChecked(False)
        crop_input_label=QLabel("px")
        crop_input_label.setStyleSheet(f"font-size: 12px; color: {COLORS['text_secondary']}; border: none;")
        self.crop_input=QLineEdit("5")
        self.crop_input.setFixedWidth(45)
        self.crop_input.setValidator(QIntValidator(0, 100))
        self.crop_input.setEnabled(False)

        crop_row=QHBoxLayout()
        crop_row.addWidget(crop_label)
        crop_row.addStretch()
        crop_row.addWidget(self.crop_check)
        crop_row.addWidget(self.crop_input)
        crop_row.addWidget(crop_input_label)

        self.crop_check.stateChanged.connect(self.onCropToggled)
        self.crop_input.editingFinished.connect(self.onCropChanged)

        #Estos toolTips son provisionales, no se si funcionan bien del todo
        vlow_label.setToolTip("Low end of the intensity range. Values outside [0,255] are clipped.")
        vhigh_label.setToolTip("High end of the intensity range. Values outside [0,255] are clipped.")
        brightness_label.setToolTip("Values outside [-127,127] are clipped.")
        contrast_label.setToolTip("Float value. Use '.' as decimal separator.")
        crop_label.setToolTip("Pixels to be croped from the image, use it if the borders could induce errors in domain separation.\nValues outside [0,100] are clipped.")

        #Proper Layout
        right_layout.addWidget(title)
        right_layout.addLayout(vlow_row)
        right_layout.addWidget(self.vlow_slider)
        right_layout.addSpacing(4)
        right_layout.addLayout(vhigh_row)
        right_layout.addWidget(self.vhigh_slider)
        right_layout.addSpacing(12)
        right_layout.addLayout(brightness_row)
        right_layout.addSpacing(4)
        right_layout.addLayout(contrast_row)
        right_layout.addSpacing(8)
        right_layout.addWidget(self.reset_but)
        right_layout.addSpacing(12)
        right_layout.addWidget(separator2)
        right_layout.addSpacing(8)
        right_layout.addLayout(crop_row)
        right_layout.addStretch()

        #Update Values
        self.vlow_slider.valueChanged.connect(self.onSliderVlow)
        self.input_vlow.editingFinished.connect(self.onInputVlow)
        self.vhigh_slider.valueChanged.connect(self.onSliderVhigh)
        self.input_vhigh.editingFinished.connect(self.onInputVhigh)
        self.input_brightness.editingFinished.connect(self.onInputBrightness)
        self.input_contrast.editingFinished.connect(self.onInputContrast)

        #Scroll
        scroll=QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        return scroll
    
    def buildBottomRow(self) -> QHBoxLayout:
        row=QHBoxLayout()
        row.addStretch()

        self.cancel_but=QPushButton("Cancel")
        self.cancel_but.setObjectName("cancel_btn")
        self.cancel_but.setFixedWidth(100)
        self.cancel_but.clicked.connect(self.cancelled)

        self.next_but=QPushButton("Next ->")
        self.next_but.setFixedWidth(100)
        self.next_but.clicked.connect(self.onNextClicked)

        row.addWidget(self.cancel_but)
        row.addSpacing(8)
        row.addWidget(self.next_but)

        return row
    
    #Update cut values
    def onSliderVlow(self, value:int) -> None:
        if self.updating: #Avoid continous updating
            return
        self.updating=True
        self.v_low=value
        self.input_vlow.setText(str(value))
        self.updating=False
        self.applyCorrection()
        self.updateHistogram()

    def onInputVlow(self) -> None:
        if self.updating:
            return
        try:
            value = max(0, min(255, int(self.input_vlow.text())))#Clip values
        except ValueError or value>self.v_high: #if not valid parameter we ignore input
            value=self.v_low
        self.updating=True
        self.v_low=value
        self.vlow_slider.setValue(value)
        self.input_vlow.setText(str(value))
        self.updating=False
        self.applyCorrection()
        self.updateHistogram()
    
    def onSliderVhigh(self, value:int) -> None:
        if self.updating: #Avoid continous updating
            return
        self.updating=True
        self.v_high=value
        self.input_vhigh.setText(str(value))
        self.updating=False
        self.applyCorrection()
        self.updateHistogram()

    def onInputVhigh(self) -> None:
        if self.updating:
            return
        try:
            value = max(0, min(255, int(self.input_vhigh.text())))#Clip values
        except ValueError or value<self.v_low: #if not valid parameter we ignore input
            value=self.v_high
        self.updating=True
        self.v_high=value
        self.vhigh_slider.setValue(value)
        self.input_vhigh.setText(str(value))
        self.updating=False
        self.applyCorrection()
        self.updateHistogram()

    def onInputBrightness(self) -> None:
        if self.updating:
            return
        try:
            value = max(-127, min(127, int(self.input_brightness.text())))
        except ValueError:
            value = self.brightness
        self.brightness = value
        self.input_brightness.setText(str(value))
        self.applyCorrection()
        self.updateHistogram()

    def onInputContrast(self) -> None:
        if self.updating:
            return
        locale = QLocale()
        text = self.input_contrast.text().replace(locale.decimalPoint(), '.')
        try:
            value = max(0.1, float(text))
        except ValueError:
            value = self.contrast
        self.contrast = value
        self.input_contrast.setText(str(round(value, 2)))
        self.applyCorrection()
        self.updateHistogram()

    def onReset(self) -> None:
        if self.image is None:
            return
        self.v_low, self.v_high = self.corrector.histogram_range(self.image, self.corrector.coverage)
        self.brightness = 0
        self.contrast = 1.0
        self.updateValues()
        self.applyCorrection()
        self.updateHistogram()
    
    def onCropToggled(self, state:int)->None:
        self.crop_enabled=bool(state)
        self.crop_input.setEnabled(self.crop_enabled)
        self.applyCorrection()
        self.updateHistogram()
    
    def onCropChanged(self)->None:
        try:
            self.crop_px=max(0,min(100,int(self.crop_input.text())))
        except ValueError:
            self.crop_px=5
        self.crop_input.setText(str(self.crop_px))
        if self.crop_enabled:
            self.applyCorrection()
            self.updateHistogram()

        
    def onNextClicked(self) -> None:
        if self.corrected is not None:
            self.correction_accepted.emit(self.corrected,self.v_low,self.v_high)

    def updateValues(self) -> None:
        self.updating = True
        self.vlow_slider.setValue(self.v_low)
        self.input_vlow.setText(str(self.v_low))
        self.vhigh_slider.setValue(self.v_high)
        self.input_vhigh.setText(str(self.v_high))
        self.input_brightness.setText(str(self.brightness))
        self.input_contrast.setText(str(self.contrast))
        self.updating = False

    def applyCorrection(self) -> None:
        if self.image is None or self.v_high == self.v_low:
            return
        image=self.image
        if self.crop_enabled and self.crop_px>0:
            n=self.crop_px
            image=image[n:-n,n:-n]
        stretched = self.corrector.linear_stretch(image, self.v_low, self.v_high)
        self.corrected = self.corrector.adjust_brightness_contrast(stretched, self.brightness, self.contrast)
        self.updateImageView()

    def updateImageView(self) -> None:
        if self.corrected is None:
            return
        height,width=self.corrected.shape
        q_image=QImage(self.corrected.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image).scaled(
            QSize(self.image_view.width() or 550, self.image_view.height() or 350),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_view.setPixmap(pixmap)

    def updateHistogram(self) -> None:
        if self.image is None:
            return
        image=self.image
        if self.crop_enabled and self.crop_px>0:
            n=self.crop_px
            image = image[n:-n,n:-n]
        self.ax.clear()
        self.ax.hist(
            image.ravel(),
            bins=256,
            range=(0,255),
            color=COLORS['accent'],
            alpha=0.75,
            linewidth=0
        )
        self.ax.axvline(
            self.v_low,
            color=COLORS['vlow'],
            linewidth=1.5,
            linestyle='--',
            label=f'Low value={self.v_low}'
        )
        self.ax.axvline(
            self.v_high,
            color=COLORS['vhigh'],
            linewidth=1.5,
            linestyle='--',
            label=f'High value={self.v_high}'
        )
        xlim_low=max(0,self.v_low-10)
        xlim_high=min(255,self.v_high+10)
        self.ax.set_xlim(xlim_low,xlim_high)
        self.ax.set_facecolor(COLORS['panel'])
        self.ax.tick_params(labelsize=8)
        self.ax.legend(fontsize=8,framealpha=0.5)
        self.figure.tight_layout()
        self.canvas.draw()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.updateImageView()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.updateImageView()


