from __future__ import annotations
import colorsys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QLabel, QSlider, QLineEdit,
    QSizePolicy,QFileDialog, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QIntValidator
from PySide6.QtCore import QThread, Qt, Signal, QSize

from core.pipeline import PipelineDictator
from interface.styles import COLORS
from core.session import Session
from processing.domainService import DomainService
from stats.domainStats import computeDomainStats
from persistence.session_io import saveSession,exportCorrectedImage
class DomainsWorker(QThread):
    finished=Signal(object)
    error=Signal(str)
    
    def __init__(self,session:Session):
        super().__init__()
        self.session=session
    
    def run(self)->None:
        try:
            domain_service=DomainService(self.session.segmentation_container)
            self.session.domain_data=domain_service.get_data()
            self.session.domain_stats=computeDomainStats(self.session.domain_data["labeled_images"])
            self.finished.emit(self.session)
        except Exception as e:
            self.error.emit(str(e))

class ResultsPanel(QWidget):
    save_requested=Signal(object)
    export_requested=Signal(object)
    cancelled=Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Session|None=None
        self.active_state: int|None=0
        self.active_metric: str=""
        self.min_area: float=0.0
        self.state_tabs: dict[int|str,QPushButton]={}
        self.metric_buttons: dict[str,QPushButton]={}
        self.xlim_min: float=0.0
        self.xlim_max: float=0.0
        self.worker: DomainsWorker|None=None
        self.extension: str=".session"
        self.buildUi()

    def buildUi(self):
        base=QVBoxLayout(self)
        base.setContentsMargins(24,26,24,16)
        base.setSpacing(12)
        base.addLayout(self.buildTopRow())

        content=QHBoxLayout()
        content.setSpacing(12)
        content.addWidget(self.buildImageFrame("Original", "original_view"), stretch=3)
        content.addWidget(self.buildImageFrame("Domains", "domain_view"), stretch=3)
        content.addWidget(self.buildHistogramFrame(), stretch=2)
        base.addLayout(content, stretch=1)

        base.addLayout(self.buildInputRow())
        base.addLayout(self.buildBottomRow())

    def buildTopRow(self)->QHBoxLayout:
        row=QHBoxLayout()
        row.setSpacing(6)
        self.tabs_container=QHBoxLayout()
        self.tabs_container.setSpacing(6)
        row.addLayout(self.tabs_container)
        row.addStretch()
        self.metrics_container=QHBoxLayout()
        self.metrics_container.setSpacing(6)
        row.addLayout(self.metrics_container)
        return row
    
    def buildImageFrame(self, title: str, attr_name:str)->QFrame:
        frame=QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"""
                QFrame {{
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    background-color: {COLORS['panel']};
                }}
            """
        )
        frame_layout=QVBoxLayout(frame)
        frame_layout.setContentsMargins(8,8,8,8)
        frame_layout.setSpacing(4)

        title_lbl=QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text_secondary']}; border: none;"
        )
        view=QLabel()
        view.setAlignment(Qt.AlignCenter)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view.setMinimumHeight(250)
        setattr(self,attr_name,view)

        frame_layout.addWidget(title_lbl)
        frame_layout.addWidget(view)
        return frame
    
    def buildHistogramFrame(self)->QFrame:
        frame=QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"""
                QFrame {{
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    background-color: {COLORS['panel']};
                }}
            """
        )
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.setSpacing(4)

        title_lbl = QLabel("Distribución")
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text_secondary']}; border: none;"
        )

        self.figure = Figure(tight_layout=True)
        self.figure.patch.set_facecolor(COLORS['panel'])
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        frame_layout.addWidget(title_lbl)
        frame_layout.addWidget(self.canvas, stretch=1)
        return frame
    
    def buildInputRow(self)->QHBoxLayout:
        row=QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self.buildAreaContainer(), stretch=1)
        row.addWidget(self.buildXlimContainer())
        return row

    def buildAreaContainer(self)->QFrame:
        frame=QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(f"QFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; background-color: {COLORS['panel']}; }}")
        layout=QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        lbl=QLabel("Minimum area")
        lbl.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']}; font-weight: bold; border: none;")

        controls=QHBoxLayout()
        self.area_slider=QSlider(Qt.Horizontal)
        self.area_slider.setRange(0,10000)
        self.area_slider.setValue(0)
        self.area_input=QLineEdit("0")
        self.area_input.setFixedWidth(64)
        self.area_input.setValidator(QIntValidator(0,9999999))
        px=QLabel("px")
        px.setStyleSheet(f"font-size: 11px; color: {COLORS['text_secondary']}; border: none;")

        self.area_slider.valueChanged.connect(self.onSliderChanged)
        self.area_input.editingFinished.connect(self.onInputChanged)

        controls.addWidget(self.area_slider)
        controls.addWidget(self.area_input)
        controls.addWidget(px)

        layout.addWidget(lbl)
        layout.addLayout(controls)
        return frame

    def buildXlimContainer(self) -> QFrame:
        frame=QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(f"QFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; background-color: {COLORS['panel']}; }}")
        layout=QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        lbl=QLabel("Histogram range")
        lbl.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']}; font-weight: bold; border: none;")

        controls=QHBoxLayout()
        controls.setSpacing(6)

        min_lbl=QLabel("Min")
        min_lbl.setStyleSheet(f"font-size: 11px; color: {COLORS['text_secondary']}; border: none;")
        self.xlim_min_input = QLineEdit("0")
        self.xlim_min_input.setFixedWidth(64)
        self.xlim_min_input.setValidator(QIntValidator(0, 9999999))

        sep=QLabel("—")
        sep.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")

        max_lbl=QLabel("Max")
        max_lbl.setStyleSheet(f"font-size: 11px; color: {COLORS['text_secondary']}; border: none;")
        self.xlim_max_input=QLineEdit("0")
        self.xlim_max_input.setFixedWidth(64)
        self.xlim_max_input.setValidator(QIntValidator(0, 9999999))

        px=QLabel("px  (0 = auto)")
        px.setStyleSheet(f"font-size: 10px; color: {COLORS['text_secondary']}; border: none;")

        self.xlim_min_input.editingFinished.connect(self.onXlimMinChanged)
        self.xlim_max_input.editingFinished.connect(self.onXlimMaxChanged)

        controls.addWidget(min_lbl)
        controls.addWidget(self.xlim_min_input)
        controls.addWidget(sep)
        controls.addWidget(max_lbl)
        controls.addWidget(self.xlim_max_input)
        controls.addWidget(px)
        controls.addStretch()

        layout.addWidget(lbl)
        layout.addLayout(controls)
        return frame
    
    def buildBottomRow(self)->QHBoxLayout:
        row_layout=QHBoxLayout()

        self.explorer_but=QPushButton("Domain Explorer")
        self.explorer_but.setFixedWidth(160)
        self.explorer_but.clicked.connect(self.onExplorerClicked)
        row_layout.addWidget(self.explorer_but)

        row_layout.addStretch()

        self.cancel_but=QPushButton("Cancelar")
        self.cancel_but.setObjectName("cancel_btn")
        self.cancel_but.setFixedWidth(100)
        self.cancel_but.clicked.connect(self.cancelled)

        self.export_but=QPushButton("Export Corrected Image")
        self.export_but.setFixedWidth(180)
        self.export_but.clicked.connect(self.onExportClicked)

        self.save_but=QPushButton("Save")
        self.save_but.setFixedWidth(100)
        self.save_but.clicked.connect(self.onSaveClicked)

        row_layout.addWidget(self.cancel_but)
        row_layout.addSpacing(8)
        row_layout.addWidget(self.export_but)
        row_layout.addSpacing(8)
        row_layout.addWidget(self.save_but)
        return row_layout

    def loadSession(self, session: Session)->None:
        self.session=session
        if session.segmentation_container is not None:
            self.worker=DomainsWorker(session)
            self.worker.finished.connect(self.onDomainsFinished)
            self.worker.error.connect(self.onDomainsError)
            self.worker.start()
        self.min_area=0.0
        self.active_state=0

        areas = []
        for state_data in session.domain_stats.values():
            for domain in state_data.values():
                areas.append(domain["area"])
        max_area = max(areas, default=10000)

        self.area_slider.setRange(0,int(max_area))
        self.area_slider.setValue(0)
        self.area_input.setText("0")

        self.buildStateTabs()
        self.buildMetricButtons()
        self.updateOriginalView()
        self.updateDomainView()
        self.updateHistogram()
    
    def buildStateTabs(self)->None:
        while self.tabs_container.count():
            item=self.tabs_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.state_tabs.clear()

        if self.session is None:
            return
        
        num_states=len(self.session.domain_data.get("labeled_images",{}))
        for state in range(num_states):
            but=QPushButton(f"Estado {state+1}")
            but.setFixedHeight(28)
            but.clicked.connect(lambda _, state=state: self.onStateTabClicked(state))
            self.state_tabs[state]=but
            self.tabs_container.addWidget(but)
        
        all_but=QPushButton("All States")
        all_but.setFixedHeight(28)
        all_but.clicked.connect(lambda: self.onStateTabClicked(None))
        self.state_tabs["all"]=all_but
        self.tabs_container.addWidget(all_but)
        self.highlightTab(0)

    def buildMetricButtons(self)->None:
        while self.metrics_container.count():
            item=self.metrics_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.metric_buttons.clear()

        if self.session is None or not self.session.domain_stats:
            return
        
        for first in self.session.domain_stats.values():
            break
        if not first:
            return
        domains_dict=list(first.values())
        first_domain=domains_dict[0]
        metrics_names=list(first_domain.keys())

        for metric in metrics_names:
            metric_but=QPushButton(metric.capitalize())
            metric_but.setFixedHeight(28)
            metric_but.clicked.connect(lambda _,metric=metric:self.onMetricClicked(metric))
            self.metric_buttons[metric]=metric_but
            self.metrics_container.addWidget(metric_but)

        if metrics_names:
            self.active_metric=metrics_names[0]
            self.highlightMetric(metrics_names[0])

    def onDomainsFinished(self, session: Session) -> None:
        self.session = session
        self.min_area = 0.0
        self.active_state = 0

        areas=[]
        for state_data in session.domain_stats.values():
            for domain in state_data.values():
                areas.append(domain["area"])
        max_area=max(areas, default=10000)

        self.area_slider.setRange(0, int(max_area))
        self.area_slider.setValue(0)
        self.area_input.setText("0")

        self.buildStateTabs()
        self.buildMetricButtons()
        self.updateOriginalView()
        self.updateDomainView()
        self.updateHistogram()

    def onDomainsError(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error calculando dominios", message)

    def onStateTabClicked(self,state: int|None)->None:
        self.active_state=state
        self.highlightTab(state if state is not None else "all")
        self.updateDomainView()
        self.updateHistogram()
    
    def onMetricClicked(self, metric: str)->None:
        self.active_metric=metric
        self.highlightMetric(metric)
        self.updateHistogram()

    def onExplorerClicked(self)->None:
        if self.session is None:
            return
        from interface.panels.domainExplorerWindow import DomainExplorerWindow
        window=DomainExplorerWindow(self.session, self.min_area, parent=self)
        window.show()
    
    def onSliderChanged(self, value: int)->None:
        self.min_area=float(value)
        self.area_input.setText(str(value))
        self.updateDomainView()
        self.updateHistogram()
    
    def onInputChanged(self)->None:
        try:
            value=max(0,int(self.area_input.text()))
        except ValueError:
            value=int(self.min_area)
        self.min_area=float(value)
        self.area_slider.setValue(value)
        self.updateDomainView()
        self.updateHistogram()

    def onXlimMinChanged(self) -> None:
        try:
            self.xlim_min=max(0,float(self.xlim_min_input.text()))
        except ValueError:
            self.xlim_min=0.0
        self.updateHistogram()

    def onXlimMaxChanged(self)->None:
        try:
            self.xlim_max=max(0,float(self.xlim_max_input.text()))
        except ValueError:
            self.xlim_max=0.0
        self.updateHistogram()

    def onSaveClicked(self)->None:
        if self.session is None:
            return
        path, _=QFileDialog.getSaveFileName(
                self, 
                "Save Session", f"{self.session.image_name}{self.extension}", 
                f"Session Files (*{self.extension})"
                )
        if not path:
            return
        try:
            saveSession(self.session, path)
            QMessageBox.information(self, "Session Saved", f"Session saved successfully to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Session", f"An error occurred while saving the session:\n{str(e)}")

    def onExportClicked(self)->None:
        if self.session is None or self.session.corrected_image is None:
            QMessageBox.warning(self, "No Corrected Image", "There is no corrected image to export.")
            return
        path, _=QFileDialog.getSaveFileName(
            self,
            "Export Corrected Image",
            f"{self.session.image_name}_corrected.tif",
            "TIFF Files (*.tif)"
        )
        if not path:
            return
        try:
            exportCorrectedImage(self.session, path)
            QMessageBox.information(self, "Image Exported", f"Corrected image exported successfully to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Exporting Image", f"An error occurred while exporting the image:\n{str(e)}")

    def highlightTab(self,active_key: int|str)->None:
        for key, but in self.state_tabs.items():
            but.setStyleSheet(self.tabStyle(key==active_key))
    
    def highlightMetric(self,active_metric: int|str)->None:
        for metric, but in self.metric_buttons.items():
            but.setStyleSheet(self.tabStyle(metric==active_metric))
    
    def tabStyle(self, active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background-color: {COLORS['accent']}; color: white; "
                f"border-radius: 5px; border: none; padding: 4px 10px; font-size: 12px; }}"
            )
        return (
            f"QPushButton {{ background-color: {COLORS['panel']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 5px; "
            f"padding: 4px 10px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: {COLORS['border']}; }}"
        )
    
    def updateOriginalView(self)->None:
        if self.session is None:
            return
        img=self.toUint8(self.session.corrected_image)
        height, width=img.shape
        img=np.ascontiguousarray(img)
        q_img=QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        pixmap=QPixmap.fromImage(q_img).scaled(
            QSize(self.original_view.width() or 400, self.original_view.height() or 300),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.original_view.setPixmap(pixmap)
    
    def updateDomainView(self)->None:
        if self.session is None:
            return
        rgb=(
            self.buildAllStatesImage()
            if self.active_state is None
            else self.buildStateImage(self.active_state)
        )
        height,width,_=rgb.shape
        rgb=np.ascontiguousarray(rgb)
        q_img=QImage(rgb.data,width,height,width*3,QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            QSize(self.domain_view.width() or 400, self.domain_view.height() or 350),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.domain_view.setPixmap(pixmap)
    
    def buildStateImage(self,state:int)->np.ndarray:
        labeled_image=self.session.domain_data["labeled_images"][state]
        stats=self.session.domain_stats.get(state,{})
        height,width=labeled_image.shape
        rgb=np.zeros((height,width,3), dtype=np.uint8)
        rng=np.random.default_rng(state*1000+42)
        domain_ids=[domain_id for domain_id, dictionary in stats.items() if dictionary["area"]>=self.min_area]
        colors=rng.integers(60,230,size=(len(domain_ids),3), dtype=np.uint8)
        for i, domain_id in enumerate(domain_ids):
            rgb[labeled_image==domain_id]=colors[i]
        return rgb
    
    def buildAllStatesImage(self)->np.ndarray:
        all_labeled_images=self.session.domain_data["labeled_images"]
        for first in all_labeled_images.values():
            break
        height,width=first.shape
        rgb=np.zeros((height,width,3), dtype=np.uint8)
        rng=np.random.default_rng(42)
        for state, labeled_image in all_labeled_images.items():
            stats=self.session.domain_stats.get(state,{})
            domain_ids=[domain_id for domain_id, dictionary in stats.items() if dictionary["area"]>=self.min_area]
            colors = rng.integers(60, 230, size=(len(domain_ids),3), dtype=np.uint8)
            for i, domain_id in enumerate(domain_ids):
                rgb[labeled_image == domain_id]=colors[i]
        return rgb

    def updateHistogram(self)->None:
        self.ax.clear()
        if self.session is None or not self.active_metric:
            self.canvas.draw()
            return
        if self.active_state is None:
            sorted_states=sorted(
                self.session.domain_stats.keys(),
                key=lambda s: self.session.segmentation_container.parameters[s]['mean']
            )
            n=len(sorted_states)
            for i,state in enumerate(sorted_states):
                values=[
                    domain[self.active_metric] for domain in self.session.domain_stats.get(state,{}).values()
                    if domain["area"]>=self.min_area
                ]
                if not values:
                    continue
                hue=240-int(i*240/max(n-1,1))
                r,g,b=colorsys.hsv_to_rgb(hue/360, 0.85, 1.0)
                color=(r,g,b)
                self.ax.hist(
                    values,
                    bins=min(30,len(values)),#mejorar con reglas euristicas para elegir el bins sturges, scott, freedman-diaconis, numero de puntos
                    color=color,
                    alpha=0.6,
                    linewidth=0,
                    label=f"Estado {state+1}"
                )
        else:
            values=self.getFilteredValues(self.active_metric)
            if values:
                self.ax.hist(
                    values,
                    bins=min(30,len(values)),
                    color=COLORS['accent'],
                    alpha=0.6,
                    linewidth=0
                )
        self.ax.set_xlabel(self.active_metric.capitalize(), fontsize=8)
        self.ax.set_ylabel("Number of domains", fontsize=8)
        self.ax.set_facecolor(COLORS['panel'])
        self.ax.tick_params(labelsize=7)
        if self.xlim_max > self.xlim_min:
            self.ax.set_xlim(self.xlim_min, self.xlim_max)
        self.canvas.draw()
    
    def getFilteredValues(self, metric: str)->list[float]:
        if self.session is None:
            return []
        states=(
            self.session.domain_stats.keys()
            if self.active_state is None
            else [self.active_state]
        )
        return[
            dictionary[metric] for state in states
            for dictionary in self.session.domain_stats.get(state,{}).values()
            if dictionary["area"]>=self.min_area
        ]
    
    def toUint8(self, image:np.ndarray)-> np.ndarray:
        img=image.astype(np.float32)
        min_value,max_value=img.min(),img.max()
        if max_value>min_value:
            img=(img-min_value)/(max_value-min_value)*255
        return img.astype(np.uint8)

    def resizeEvent(self, event)->None:
        super().resizeEvent(event)
        self.updateOriginalView()
        self.updateDomainView()

    def showEvent(self, event)->None:
        super().showEvent(event)
        self.updateOriginalView()
        self.updateDomainView()



