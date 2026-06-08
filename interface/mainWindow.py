from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QMainWindow, QStackedWidget
from PySide6.QtCore import Qt

from core.session import Session
from interface.panels.loadPanel import LoadPanel
from interface.panels.bcPanel import BcPanel
from interface.panels.icmPanel import IsingPanel
from interface.panels.resultsPanel import ResultsPanel
from interface.styles import app_stylesheet
from core.segmentationContainer import SegmentationContainer

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Domain Analysis Tool")
        self.setMinimumSize(1000,700)
        self.setStyleSheet(app_stylesheet())

        self.image: np.ndarray|None=None
        self.image_name: str=""

        self.corrected_image: np.ndarray | None = None

        self.buildUi()

    def buildUi(self):
        self.stack=QStackedWidget()
        self.setCentralWidget(self.stack)

        self.load_panel=LoadPanel()
        self.load_panel.image_loaded.connect(self.onImageLoaded)
        self.load_panel.next_but.clicked.connect(self.goToBc)
        self.load_panel.session_loaded.connect(self.onSessionLoaded)
        self.stack.addWidget(self.load_panel)

        self.bc_panel=BcPanel()
        self.bc_panel.correction_accepted.connect(self.onCorrectionAccepted)
        self.bc_panel.cancelled.connect(self.goToLoad)
        self.stack.addWidget(self.bc_panel)

        self.ising_panel = IsingPanel()
        self.ising_panel.segmentation_accepted.connect(self.onSegmentationAccepted)
        self.ising_panel.cancelled.connect(self.goToBc)
        self.stack.addWidget(self.ising_panel)

        self.results_panel = ResultsPanel()
        self.results_panel.cancelled.connect(self.goToIsing)
        self.stack.addWidget(self.results_panel)

    def onImageLoaded(self, image:np.ndarray, name: str):
        self.image = image
        self.image_name= name
    
    def onSessionLoaded(self, session:Session)->None:
        self.session=session
        self.results_panel.cancelled.disconnect()
        self.results_panel.cancelled.connect(self.goToLoad)
        self.results_panel.loadSession(session)
        self.stack.setCurrentIndex(3)

    def goToBc(self):
        if self.image is not None:
            self.bc_panel.defaultCorrection(self.image)
            self.stack.setCurrentIndex(1)

    def goToLoad(self):
        self.stack.setCurrentIndex(0)

    def onCorrectionAccepted(self, corrected: np.ndarray, v_low: int, v_high: int):
        self.corrected_image = corrected
        self.ising_panel.loadImage(corrected)
        self.stack.setCurrentIndex(2)

    def goToIsing(self):
        self.stack.setCurrentIndex(2)

    def onSegmentationAccepted(self, ising_container: SegmentationContainer):
        from core.pipeline import PipelineDictator
        self.session = Session(
            image_name=self.image_name,
            original_image=self.image,
            corrected_image=self.corrected_image,
            ising_result=ising_container.final_image,
            parameters=ising_container.method_configuration,
            ising_stats=ising_container.parameters,
            segmentation_container=ising_container,
            segmentation_method=ising_container.method
        )
        pipeline = PipelineDictator()
        self.results_panel.cancelled.disconnect()
        self.results_panel.cancelled.connect(self.goToIsing)
        self.session = pipeline.run_domains(self.session)
        self.results_panel.loadSession(self.session)
        self.stack.setCurrentIndex(3)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec())
