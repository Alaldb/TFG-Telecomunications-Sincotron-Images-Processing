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
from interface.panels.graphCutsPanel import GraphCutsPanel
from interface.panels.startPanel import StartPanel
from interface.panels.domainComparisonPanel import DomainComparisonPanel

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

        self.start_panel = StartPanel()
        self.start_panel.analyse_requested.connect(self.goToLoad)
        self.start_panel.compare_requested.connect(self.goToComparison)
        self.stack.addWidget(self.start_panel)

        self.load_panel=LoadPanel()
        self.load_panel.image_loaded.connect(self.onImageLoaded)
        self.load_panel.next_but.clicked.connect(self.goToBc)
        self.load_panel.home.connect(self.goToStart)
        self.load_panel.session_loaded.connect(self.onSessionLoaded)
        self.stack.addWidget(self.load_panel)

        self.bc_panel=BcPanel()
        self.bc_panel.correction_accepted.connect(self.onCorrectionAccepted)
        self.bc_panel.cancelled.connect(self.goToLoad)
        self.bc_panel.home.connect(self.goToStart)
        self.stack.addWidget(self.bc_panel)

        self.imc_panel = IsingPanel()
        self.imc_panel.segmentation_accepted.connect(self.onSegmentationAccepted)
        self.imc_panel.cancelled.connect(self.goToBc)
        self.imc_panel.home.connect(self.goToStart)
        self.stack.addWidget(self.imc_panel)

        self.graph_cuts_panel = GraphCutsPanel()
        self.graph_cuts_panel.segmentation_accepted.connect(self.onSegmentationAccepted)
        self.graph_cuts_panel.cancelled.connect(self.goToBc)
        self.graph_cuts_panel.home.connect(self.goToStart)
        self.stack.addWidget(self.graph_cuts_panel)

        self.results_panel = ResultsPanel()
        self.results_panel.cancelled.connect(self.goToSegmentation)
        self.results_panel.home.connect(self.goToStart)
        self.stack.addWidget(self.results_panel)

        self.comparison_panel = DomainComparisonPanel()
        self.comparison_panel.home.connect(self.goToStart)
        self.stack.addWidget(self.comparison_panel)

    def onImageLoaded(self, image:np.ndarray, name: str):
        self.image = image
        self.image_name= name
    
    def onSessionLoaded(self, session:Session)->None:
        self.session=session
        self.results_panel.cancelled.disconnect()
        self.results_panel.cancelled.connect(self.goToLoad)
        self.results_panel.loadSession(session)
        self.stack.setCurrentWidget(self.results_panel)

    def goToBc(self):
        if self.image is not None:
            self.bc_panel.defaultCorrection(self.image)
            self.stack.setCurrentWidget(self.bc_panel)

    def goToLoad(self):
        self.stack.setCurrentWidget(self.load_panel)
    
    def goToComparison(self):
        self.stack.setCurrentWidget(self.comparison_panel)

    def onCorrectionAccepted(self, corrected: np.ndarray, v_low: int, v_high: int):
        self.corrected_image = corrected
        if self.load_panel.selected_method=="ICM":
            self.imc_panel.loadImage(corrected)
            self.stack.setCurrentWidget(self.imc_panel) 
        else:
            self.graph_cuts_panel.loadImage(corrected)
            self.stack.setCurrentWidget(self.graph_cuts_panel)

    def goToSegmentation(self):
        if self.load_panel.selected_method=="ICM":
            self.stack.setCurrentWidget(self.imc_panel)
        else:
            self.stack.setCurrentWidget(self.graph_cuts_panel)

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
        self.results_panel.cancelled.connect(self.goToSegmentation)
        self.session = pipeline.run_domains(self.session)
        self.results_panel.loadSession(self.session)
        self.stack.setCurrentWidget(self.results_panel)

    def goToStart(self):
        self.stack.setCurrentWidget(self.start_panel)
    


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec())
