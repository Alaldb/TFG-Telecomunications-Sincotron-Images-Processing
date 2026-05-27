from __future__ import annotations

import copy
import colorsys
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QLabel, QLineEdit,
    QSizePolicy, QMessageBox,
)
from PySide6.QtGui import QPixmap, QImage, QIntValidator
from PySide6.QtCore import Qt, Signal, QSize, QThread

from interface.styles import COLORS
from processing.isingMethodService import Ising


class IsingWorker(QThread):
    finished=Signal(np.ndarray, object)
    error=Signal(str)

    def __init__(self, image:np.ndarray, beta: float, max_iterations: int, num_states: int):
        super().__init__()
        self.image=image.copy()
        self.beta=beta
        self.max_iterations=max_iterations
        self.num_states=num_states

    def run(self) -> None:
        try:
            ising=Ising(self.beta,self.max_iterations,self.num_states)
            ising.run(self.image)
            self.finished.emit(
                ising.final_image.copy(),
                ising.parameters.copy()
            )
        except Exception as e:
            self.error.emit(str(e))

class IsingPanel(QWidget):
    segmentation_accepted=Signal(np.ndarray, object)
    cancelled=Signal()

    PARAMETERS_INFO={
        "Beta": (
            "Spatial regularization parameter.\n\n"
            "Higher values produce smoother region boundaries "
            "by increasing neighbour influence.\n"
            "Typical range: 0.5 – 5."
        ),
        "Max Iterations": (
            "Maximum number of ICM iterations.\n\n"
            "The algorithm stops earlier if it converges. "
            "More iterations allow finer refinement at the cost of speed."
        ),
        "Num States": (
            "Number of distinct intensity states to segment.\n\n"
            "3 is typical for magnetic domain images: "
            "dark domains, bright domains and intermediate state."
        ),
    }

    def __init__(self, parent=None)->None:
        super().__init__(parent)
        self.image: np.ndarray|None=None
        self.result: np.ndarray|None=None
        self.parameters: dict={}
        self.state_colors: dict={}
        self.sorted_states: list=[]
        self.active_state: int=-1
        self.worker: IsingWorker|None=None
        self.beta: float=2.0
        self.max_iterations: int=10
        self.num_states: int=3
        self.buildUi()

    def loadImage(self, image:np.ndarray)->None:
        self.image=image
        self.updateOriginalView()
        self.runIsing()
    
    def buildUi(self)->None:
        base_layout=QVBoxLayout(self)
        base_layout.setContentsMargins(24,24,24,24)
        base_layout.setSpacing(12)

        #Images+Parameters+State button
        top_row=QHBoxLayout()
        top_row.setSpacing(10)
        top_row.addWidget(self.buildImageFrame("Original","orig_view"), stretch=1)
        top_row.addWidget(self.buildImageFrame("State","state_view"), stretch=1)
        top_row.addWidget(self.buildRightPanel())

        #Histogram+legend
        middle_row=QHBoxLayout()
        middle_row.setSpacing(10)
        middle_row.addWidget(self.buildHistogramFrame(),stretch=1)
        middle_row.addWidget(self.buildLegendFrame())

        base_layout.addLayout(top_row, stretch=1)
        base_layout.addLayout(middle_row, stretch=0)
        base_layout.addLayout(self.buildBottomRow())
    
    def buildImageFrame(self, title:str, public_name:str)->QFrame:
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

        frame_view = QLabel()
        frame_view.setAlignment(Qt.AlignCenter)
        frame_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frame_view.setMinimumHeight(250)
        setattr(self, public_name, frame_view)

        frame_layout.addWidget(title_lbl)
        frame_layout.addWidget(frame_view)
        return frame

    def buildRightPanel(self)->QFrame:
        container=QFrame()
        container.setFixedWidth(180)
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet(
            f"""
                QFrame {{
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    background-color: {COLORS['panel']};
                }}
            """
        )
        right_layout=QVBoxLayout(container)
        right_layout.setContentsMargins(12,16,12,16)
        right_layout.setSpacing(10)

        title_lbl=QLabel("Ising Parameters")
        title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {COLORS['text']}; border: none;"
        )

        self.beta_input=QLineEdit(str(self.beta))
        self.beta_input.setFixedWidth(55)

        self.iterations_input=QLineEdit(str(self.max_iterations))
        self.iterations_input.setFixedWidth(55)
        self.iterations_input.setValidator(QIntValidator(1,200))

        self.states_input = QLineEdit(str(self.num_states))
        self.states_input.setFixedWidth(55)
        self.states_input.setValidator(QIntValidator(2, 8))

        self.run_but=QPushButton("Run")
        self.run_but.clicked.connect(self.runIsing)

        separator=QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"color: {COLORS['border']}; border: none; border-top: 1px solid {COLORS['border']};")

        self.buts_lbl=QLabel("States")
        self.buts_lbl.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {COLORS['text_secondary']}; border: none;"
        )
        self.buts_widget = QWidget()
        self.buts_layout = QVBoxLayout(self.buts_widget)
        self.buts_layout.setContentsMargins(0, 0, 0, 0)
        self.buts_layout.setSpacing(5)

        right_layout.addWidget(title_lbl)
        right_layout.addLayout(self.buildParamRow("Beta", self.beta_input))
        right_layout.addLayout(self.buildParamRow("Max Iterations", self.iterations_input))
        right_layout.addLayout(self.buildParamRow("Num States", self.states_input))
        right_layout.addWidget(self.run_but)
        right_layout.addWidget(separator)
        right_layout.addWidget(self.buts_lbl)
        right_layout.addWidget(self.buts_widget)
        right_layout.addStretch()

        return container
    
    def buildParamRow(self, name:str, input_widget:QLineEdit)->QHBoxLayout:
        param_row=QHBoxLayout()
        param_row.setSpacing(4)

        label=QLabel(name)
        label.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {COLORS['text']}; border: none;"
        )
        help_but=QPushButton("?")
        help_but.setFixedSize(16,16)
        help_but.setStyleSheet(
            f"""
                QPushButton {{
                    background-color: {COLORS['border']};
                    color: {COLORS['text']};
                    border-radius: 8px;
                    font-size: 10px;
                    font-weight: bold;
                    padding: 0px;
                    border: none;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['accent']};
                    color: white;
                }}
            """
        )
        help_text=self.PARAMETERS_INFO.get(name,"")
        help_but.clicked.connect(
            lambda _=False, param=name, message=help_text: QMessageBox.information(self,param, message)
        )
        param_row.addWidget(label)
        param_row.addWidget(help_but)
        param_row.addStretch()
        param_row.addWidget(input_widget)
        return param_row
    
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

        frame_layout=QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.setSpacing(4)

        title_lbl=QLabel("Pixel intensity distribution")
        title_lbl.setStyleSheet(
            f"font-size: 10px; color: {COLORS['text_secondary']}; border: none;"
        )

        self.figure=Figure(figsize=(6, 2), tight_layout=True)
        self.figure.patch.set_facecolor(COLORS['panel'])
        self.ax=self.figure.add_subplot(111)
        self.ax.set_facecolor(COLORS['panel'])
        self.ax.tick_params(labelsize=7)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedHeight(120)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        frame_layout.addWidget(title_lbl)
        frame_layout.addWidget(self.canvas)
        return frame
    
    def buildLegendFrame(self)->QFrame:
        frame=QFrame()
        frame.setFixedWidth(180)
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
        frame_layout.setContentsMargins(12, 10, 12, 10)
        frame_layout.setSpacing(6)

        title_lbl=QLabel("State distribution")
        title_lbl.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {COLORS['text_secondary']}; border: none;"
        )

        self.legend_container=QWidget()
        self.legend_layout=QVBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(0,0,0,0)
        self.legend_layout.setSpacing(5)

        frame_layout.addWidget(title_lbl)
        frame_layout.addWidget(self.legend_container)
        frame_layout.addStretch()
        return frame
    
    def buildBottomRow(self)->QHBoxLayout:
        row=QHBoxLayout()
        row.addStretch()

        self.cancel_but=QPushButton("Cancel")
        self.cancel_but.setObjectName("cancel_btn")
        self.cancel_but.setFixedWidth(100)
        self.cancel_but.clicked.connect(self.cancelled)

        self.next_but=QPushButton("Next ->")
        self.next_but.setFixedWidth(100)
        self.next_but.setEnabled(False)
        self.next_but.clicked.connect(self.onNextClicked)

        row.addWidget(self.cancel_but)
        row.addSpacing(8)
        row.addWidget(self.next_but)
        return row
    
    def runIsing(self)->None:
        if self.image is None:
            return
        self.beta=self.readBeta()
        self.max_iterations=self.readMaxIter()
        self.num_states=self.readNumStates()

        self.run_but.setEnabled(False)
        self.run_but.setText("Running...")
        self.next_but.setEnabled(False)

        self.worker=IsingWorker(self.image,self.beta,self.max_iterations,self.num_states)
        self.worker.finished.connect(self.onIsingFinished)
        self.worker.error.connect(self.onIsingError)
        self.worker.start()

    def onIsingFinished(self, result: np.ndarray, parameters: dict)->None:
        self.result=result
        self.parameters=parameters
        self.computeStateColors()
        self.rebuildStateButtons()
        self.updateStateView(self.sorted_states[0])
        self.updateHistogram()
        self.updateLegend()
        self.run_but.setEnabled(True)
        self.run_but.setText("Run")
        self.next_but.setEnabled(True)

    def onIsingError(self, message:str)->None:
        self.run_but.setEnabled(True)
        self.run_but.setText("Run")
        QMessageBox.critical(self, "Ising Error", message)

    def computeStateColors(self)->None:
        self.sorted_states=sorted(
            self.parameters.keys(),
            key=lambda parameters_index: self.parameters[parameters_index]['mean']
        )
        n=len(self.sorted_states)
        self.state_colors={}
        for i,state in enumerate(self.sorted_states):
            hue=240-int(i*240/max(n-1,1))
            r,g,b=colorsys.hsv_to_rgb(hue/360,0.75,0.85)
            self.state_colors[state]=(int(r*255),int(g*255),int(b*255))
    
    def rebuildStateButtons(self)->None:
        while self.buts_layout.count():
            item=self.buts_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i,state in enumerate(self.sorted_states):
            r,g,b=self.state_colors[state]
            lum=0.299*r+0.587*g+0.114*b
            text_color="white" if lum<160 else COLORS['text']
            button=QPushButton(f"State {i+1}")
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setStyleSheet(
                f"""
                    QPushButton {{
                        background-color: rgb({r},{g},{b});
                        color: {text_color};
                        border-radius: 5px;
                        font-weight: bold;
                        font-size: 11px;
                        border: none;
                        padding: 5px;
                    }}
                    QPushButton:hover {{
                        background-color: rgb({min(r+25,255)},{min(g+25,255)},{min(b+25,255)});
                    }}
                """
            )
            button.clicked.connect(lambda _=False, current_state=state: self.updateStateView(current_state))
            self.buts_layout.addWidget(button)
        
    def updateOriginalView(self)->None:
        if self.image is None:
            return
        height, width   = self.image.shape
        img = np.ascontiguousarray(self.image)
        q_img = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        pixmap=QPixmap.fromImage(q_img).scaled(
            QSize(self.orig_view.width() or 400, self.orig_view.height() or 300),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.orig_view.setPixmap(pixmap)
    
    def updateStateView(self, state:int)->None:
        if self.result is None or state not in self.state_colors:
            return
        self.active_state=state
        height,width=self.result.shape
        colored_image_matrix=np.zeros((height,width,3),dtype=np.uint8)
        r,g,b=self.state_colors[state]
        colored_image_matrix[self.result==state]=(r,g,b)
        colored_image_matrix=np.ascontiguousarray(colored_image_matrix)
        q_img=QImage(colored_image_matrix.data, width, height, width*3, QImage.Format_RGB888)
        pixmap=QPixmap.fromImage(q_img).scaled(
            QSize(self.state_view.width() or 400, self.state_view.height() or 300),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.state_view.setPixmap(pixmap)

    def updateHistogram(self)->None:
        if self.result is None or self.image is None:
            return
        self.ax.clear()
        for state in self.sorted_states:
            pixels=self.image[self.result==state].astype(np.float32)
            if len(pixels)<2:
                continue
            r,g,b=self.state_colors[state]
            color=(r/255,g/255,b/255)
            self.ax.hist(
                pixels,
                bins=100,
                range=(0,255),
                density=True,
                color=color,
                alpha=0.4,
                linewidth=0
            )
            #recomendación IA para reducir el tiempo de creaión de histograma, en vez de coger todos los píxeles se selecciona solo una muestra
            sample=(
                pixels if len(pixels)<=50_000
                else np.random.choice(pixels,50_000,replace=False)
            )
            if sample.std() < 1e-6:
                continue
            kde=gaussian_kde(sample)
            x=np.linspace(0,255,500)
            self.ax.plot(x,kde(x),color=color,linewidth=1.5)

        self.ax.set_xlim(0,255)
        self.ax.set_xlabel("Pixel intensity", fontsize=7)
        self.ax.set_ylabel("Density", fontsize=7)
        self.ax.set_facecolor(COLORS['panel'])
        self.ax.tick_params(labelsize=7)
        self.ax.grid(linestyle='--', alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def updateLegend(self)->None:
        while self.legend_layout.count():
            item=self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        total=int((self.result>=0).sum())
        for i,state in enumerate(self.sorted_states):
            count=int((self.result==state).sum())
            percent=100*count/total if total>0 else 0.0
            r,g,b=self.state_colors[state]

            row_widget=QWidget()
            row_layout=QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0,0,0,0)
            row_layout.setSpacing(6)

            color_sample=QLabel()
            color_sample.setFixedSize(12,12)
            color_sample.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border-radius: 3px; border: none;"
            )

            text=QLabel(f"Estado {i+1}: {percent:.2f}%")
            text.setStyleSheet(
                f"font-size: 11px; color: {COLORS['text']}; border: none;"
            )
            row_layout.addWidget(color_sample)
            row_layout.addWidget(text)
            row_layout.addStretch()
            self.legend_layout.addWidget(row_widget)

    def readBeta(self)->float:
        try:
            return max(0.0, float(self.beta_input.text().replace(',', '.')))
        except ValueError:
            return self.beta

    def readMaxIter(self) -> int:
        try:
            return max(1, min(200, int(self.iterations_input.text())))
        except ValueError:
            return self.max_iterations
        
    def readNumStates(self) -> int:
        try:
            return max(2, min(8, int(self.states_input.text())))
        except ValueError:
            return self.num_states
            
        
    def onNextClicked(self) -> None:
        if self.result is not None:
            self.segmentation_accepted.emit(self.result, self.parameters)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.updateOriginalView()
        self.updateStateView(self.active_state)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.updateOriginalView()
        self.updateStateView(self.active_state)


