from __future__ import annotations
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QLabel, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from interface.styles import COLORS
from interface.visual_elements.domainImageViewer import DomainImageViewer
from persistence.session_io import loadSession
from core.session import Session
from processing.domainComparisonService import DomainComparisonService
from interface.panels.comparisonResultsWindow import ComparisonResultsWindow

class DomainComparisonPanel(QWidget):
    home=Signal()

    def __init__(self, parent=None)->None:
        super().__init__(parent)
        self.session_a: Session|None=None
        self.session_b: Session|None=None
        self.active_state_a: int=0
        self.active_state_b: int=0
        self.selected_domain_a: int|None=None
        self.selected_domain_b: int|None=None
        self.state_tabs_a: dict[int, QPushButton]={}
        self.state_tabs_b: dict[int, QPushButton]={}
        self.buildUi()
    
    def buildUi(self)->None:
        base=QVBoxLayout(self)
        base.setContentsMargins(24,20,24,16)
        base.setSpacing(12)

        viewers_row=QHBoxLayout()
        viewers_row.setSpacing(12)
        viewers_row.addWidget(self.buildSessionView("A"))
        viewers_row.addWidget(self.buildSessionView("B"))

        base.addLayout(viewers_row, stretch=1)
        base.addLayout(self.buildBottomRow())
    
    def buildSessionView(self, session_id:str)->QFrame:
        frame=QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; background-color: {COLORS['panel']}; }}"
        )
        session_layout=QVBoxLayout(frame)
        session_layout.setContentsMargins(12,12,12,12)
        session_layout.setSpacing(8)

        top_row=QHBoxLayout()

        load_but=QPushButton(f"Load Session {session_id}")
        load_but.setFixedHeight(28)
        load_but.clicked.connect(lambda: self.onLoadSession(session_id))

        label=QLabel("No session loaded")
        label.setStyleSheet(f"font-size: 12px; color: {COLORS['text_secondary']}; border:none;")

        top_row.addWidget(load_but)
        top_row.addWidget(label)
        top_row.addStretch()

        state_row=QHBoxLayout()
        state_row.setSpacing(6)

        viewer=DomainImageViewer()
        viewer.domain_clicked.connect(lambda domain_id, metrics: self.onDomainClicked(session_id, domain_id, metrics))

        session_layout.addLayout(top_row)
        session_layout.addLayout(state_row)
        session_layout.addWidget(viewer,stretch=1)

        if session_id=="A":
            self.label_a=label
            self.state_row_a=state_row
            self.viewer_a=viewer
        if session_id=="B":
            self.label_b=label
            self.state_row_b=state_row
            self.viewer_b=viewer
        
        return frame
    
    def buildBottomRow(self)->QHBoxLayout:
        row=QHBoxLayout()

        self.home_but = QPushButton("Home")
        self.home_but.setFixedWidth(120)
        self.home_but.setObjectName("cancel_btn")
        self.home_but.clicked.connect(self.home)
        row.addWidget(self.home_but)

        self.compare_but=QPushButton("Compare")
        self.compare_but.setFixedWidth(120)
        self.compare_but.setEnabled(False)
        self.compare_but.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS['border']}; color: {COLORS['text_secondary']}; "
            f"border-radius: 6px; border: none; padding: 8px 20px; font-size: 13px; }}"
        )
        self.compare_but.clicked.connect(self.onCompareClicked)

        row.addStretch()
        row.addWidget(self.compare_but)

        return row
    
    def onLoadSession(self, session_id:str)->None:
        path,_=QFileDialog.getOpenFileName(
            self, f"Load Session {session_id}", "", "Session files (*.session)"
        )
        if not path:
            return
        try:
            session=loadSession(path)
            original=session.corrected_image if session.corrected_image is not None else session.original_image
            domain_data_dict={}
            for state, labeled in session.domain_data["labeled_images"].items():
                domain_data_dict[state]={}
                for domain_id in range(1, int(labeled.max()) + 1):
                    mask=labeled==domain_id
                    domain_data_dict[state][domain_id] = {
                        "coords": np.argwhere(mask),
                        "values": original[mask] if original is not None else np.array([]),
                    }
            session.domain_data["domain_data"]=domain_data_dict
        except Exception as e:
            QMessageBox.critical(self, "Error loading session", str(e))
            return
        
        name=Path(path).name

        if session_id=="A":
            self.session_a=session
            self.selected_domain_a=None
            self.label_a.setText(name)
        if session_id=="B":
            self.session_b=session
            self.selected_domain_b=None
            self.label_b.setText(name)
        
        self.buildStateTabs(session_id)
        self.updateViewer(session_id)
        self.updateCompareButton()
    
    def buildStateTabs(self, session_id:str)->None:
        if session_id=="A":
            state_row=self.state_row_a
            state_tabs=self.state_tabs_a
            session=self.session_a
        if session_id=="B":
            state_row=self.state_row_b
            state_tabs=self.state_tabs_b
            session=self.session_b
        
        while state_row.count():
            item=state_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        state_tabs.clear()
        if session is None:
            return
        
        num_states=len(session.domain_data.get("labeled_images",{}))
        for state in range(num_states):
            state_but=QPushButton(f"State {state+1}")
            state_but.setFixedHeight(26)
            state_but.clicked.connect(lambda _, state=state: self.onStateTabClicked(session_id,state))
            state_tabs[state]=state_but
            state_row.addWidget(state_but)
        state_row.addStretch()
        self.highlightSelectedState(session_id,0)

    def highlightSelectedState(self, session_id:str, active_state:int)->None:
        state_tabs=self.state_tabs_a if session_id=="A" else self.state_tabs_b
        for state, but in state_tabs.items():
            is_active= state==active_state
            if is_active:
                but.setStyleSheet(
                    f"QPushButton {{ background-color: {COLORS['accent']}; color: white; "
                    f"border-radius: 5px; border: none; padding: 4px 10px; font-size: 12px; }}"
                )
            else:
                but.setStyleSheet(
                    f"QPushButton {{ background-color: {COLORS['panel']}; color: {COLORS['text']}; "
                    f"border: 1px solid {COLORS['border']}; border-radius: 5px; "
                    f"padding: 4px 10px; font-size: 12px; }}"
                )

    def updateViewer(self, session_id:str)->None:
        if session_id=="A":
            session=self.session_a
            state=self.active_state_a
            viewer=self.viewer_a
        if session_id=="B":
            session=self.session_b
            state=self.active_state_b
            viewer=self.viewer_b
        if session is None:
            return
        
        labeled_image=session.domain_data["labeled_images"][state]
        stats=session.domain_stats.get(state,{})

        height,width=labeled_image.shape
        rgb=np.zeros((height,width,3), dtype=np.uint8)
        rng=np.random.default_rng(state*1000+42)
        domain_ids=list(stats.keys())
        colors=rng.integers(60, 230, size=(len(domain_ids), 3), dtype=np.uint8)
        for i, domain_id in enumerate(domain_ids):
            rgb[labeled_image==domain_id]=colors[i]
        viewer.setData(rgb, labeled_image, stats)
    
    def onStateTabClicked(self, session_id:str, state:int)->None:
        if session_id=="A":
            self.active_state_a=state
            self.selected_domain_a=None
        else:
            self.active_state_b=state
            self.selected_domain_b=None
        self.highlightSelectedState(session_id,state)
        self.updateViewer(session_id)
        self.updateCompareButton()
    
    def onDomainClicked(self, session_id:str, domain_id:int, metrics:dict)->None:
        if session_id=="A":
            self.selected_domain_a=domain_id
        else:
            self.selected_domain_b=domain_id
        self.updateCompareButton()
    
    def updateCompareButton(self)->None:
        ready=self.selected_domain_a is not None and self.selected_domain_b is not None
        self.compare_but.setEnabled(ready)
        if ready:
            self.compare_but.setStyleSheet(
                f"QPushButton {{ background-color: {COLORS['accent']}; color: white; "
                f"border-radius: 6px; border: none; padding: 8px 20px; font-size: 13px; }}"
            )
        else:
            self.compare_but.setStyleSheet(
                f"QPushButton {{ background-color: {COLORS['border']}; color: {COLORS['text_secondary']}; "
                f"border-radius: 6px; border: none; padding: 8px 20px; font-size: 13px; }}"
            )
        
    def onCompareClicked(self)->None:
        state_a=self.active_state_a
        state_b=self.active_state_b

        domain_a={
            "stats": self.session_a.domain_stats[state_a][self.selected_domain_a],
            "coords": self.session_a.domain_data["domain_data"][state_a][self.selected_domain_a]["coords"],
            "values": self.session_a.domain_data["domain_data"][state_a][self.selected_domain_a]["values"],
        }
        domain_b={
            "stats": self.session_b.domain_stats[state_b][self.selected_domain_b],
            "coords": self.session_b.domain_data["domain_data"][state_b][self.selected_domain_b]["coords"],
            "values": self.session_b.domain_data["domain_data"][state_b][self.selected_domain_b]["values"],
        }

        image_shape_a=self.session_a.original_image.shape[:2]
        image_shape_b=self.session_b.original_image.shape[:2]
        image_shape = (max(image_shape_a[0], image_shape_b[0]), max(image_shape_a[1], image_shape_b[1]))#No creo que pase pero esto es por si acaso las imagenes no tienen las mismas proporciones que no se rompa.
        result=DomainComparisonService(domain_a,domain_b,image_shape).compare()
        self.results_window=ComparisonResultsWindow(result, parent=self)
        self.results_window.show()







