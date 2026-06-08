from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QLabel, QSizePolicy,
)
from PySide6.QtCore import Qt

from interface.styles import COLORS
from interface.visual_elements.domainImageViewer import DomainImageViewer
from core.session import Session

class DomainExplorerWindow(QDialog):
    def __init__(self, session: Session, min_area:float, parent=None)->None:
        super().__init__(parent)
        self.setWindowTitle("Domain Data Explorer")
        self.setMinimumSize(900,500)

        self.session=session
        self.min_area=min_area
        self.active_state: int=0
        self.selected_domain_id: int|None=None
        self.selected_domain_metrics: dict={}
        self.state_tabs: dict[int,QPushButton]={}
        self.buildUi()
        self.loadData()

    def loadData(self) -> None:
        self.highlightTab(0)
        self.updateViewer()
        self.updateMetricsPanel()

    def buildUi(self)->None:
        base=QVBoxLayout(self)
        base.setContentsMargins(20,16,20,16)
        base.setSpacing(12)
        base.addLayout(self.buildTopRow())

        content=QHBoxLayout()
        content.setSpacing(12)

        self.viewer=DomainImageViewer()
        self.viewer.domain_clicked.connect(self.onDomainClicked)
        content.addWidget(self.viewer,stretch=3)
        content.addWidget(self.buildMetricsPanel(), stretch=3)

        base.addLayout(content, stretch=1)

    def buildTopRow(self)->QHBoxLayout:
        row=QHBoxLayout()
        row.setSpacing(6)
        self.tabs_container=QHBoxLayout()
        self.tabs_container.setSpacing(6)

        num_states=len(self.session.domain_data.get("labeled_images", {}))
        for state in range(num_states):
            but=QPushButton(f"State {state+1}")
            but.setFixedHeight(28)
            but.clicked.connect(lambda _, state=state: self.onStateTabClicked(state))
            self.state_tabs[state]=but
            self.tabs_container.addWidget(but)
        row.addLayout(self.tabs_container)
        row.addStretch()
        return row
    
    def highlightTab(self, active_key: int)->None:
        for key,but in self.state_tabs.items():
            but.setStyleSheet(self.tabStyle(key==active_key))
    
    def tabStyle(self, active: bool)->str:
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
    
    def buildMetricsPanel(self)->QFrame:
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
        frame_layout.setContentsMargins(12,12,12,12)
        frame_layout.setSpacing(8)

        title=QLabel("Metrics")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 12px; font-weight: bold; color: {COLORS['text_secondary']}; border: none;"
        )

        self.domain_badge=QLabel("")
        self.domain_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.domain_badge.setStyleSheet("border: none;")

        self.metrics_table=QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Metric","Value","State mean","Global mean", "Percentage"]
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)#dinamyc number of states
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)#only read no edit
        self.metrics_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)#only read no feedback
        self.metrics_table.setStyleSheet("border: none;")#no border
        self.metrics_table.hide()

        self.empty_state_label=QLabel("Click on a domain to see its data")
        self.empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_label.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text_secondary']}; border: none;"
        )

        frame_layout.addWidget(title)
        frame_layout.addWidget(self.domain_badge)
        frame_layout.addWidget(self.metrics_table)
        frame_layout.addWidget(self.empty_state_label)
        frame_layout.addStretch()
        return frame
    
    def onStateTabClicked(self, state:int)->None:
        self.active_state=state
        self.selected_domain_id=None
        self.selected_domain_metrics={}
        self.highlightTab(state)
        self.updateViewer()
        self.updateMetricsPanel()

    def onDomainClicked(self, domain_id:int, metrics:dict)->None:
        self.selected_domain_id=domain_id
        self.selected_domain_metrics=metrics
        self.updateMetricsPanel()
    
    def updateViewer(self)->None:
        labeled=self.session.domain_data["labeled_images"][self.active_state]
        stats=self.session.domain_stats.get(self.active_state,{})

        height,width=labeled.shape
        rgb = np.zeros((height,width,3), dtype=np.uint8)
        rng=np.random.default_rng(self.active_state*1000+42)

        domain_ids=[domain_id for domain_id, dictionary in stats.items() if dictionary["area"]>=self.min_area]
        colors = rng.integers(60,230,size=(len(domain_ids),3),dtype=np.uint8)
        for i, domain_id in enumerate(domain_ids):
            rgb[labeled==domain_id]=colors[i]
        domain_stats_filtered={domain_id: stats[domain_id] for domain_id in domain_ids}
        self.viewer.setData(rgb, labeled, domain_stats_filtered)

    def updateMetricsPanel(self)->None:
        if self.selected_domain_id is None:
            self.domain_badge.setText("")
            self.metrics_table.hide()
            self.empty_state_label.show()
            return
        
        self.empty_state_label.hide()
        self.metrics_table.show()
        self.domain_badge.setText(f"Domain {self.selected_domain_id}")

        metrics=self.selected_domain_metrics
        metric_names=list(metrics.keys())
        state_means, global_means = self.computeMeans(metric_names)

        self.metrics_table.setRowCount(len(metric_names))#Dinamyc number of stats
        for row,metric in enumerate(metric_names):
            value=metrics[metric]
            state_mean=state_means[metric]
            global_mean=global_means[metric]
            percentage=((value-state_mean)/state_mean*100) if state_mean!=0 else 0
            sign="+" if percentage>=0 else ""
            color=Qt.GlobalColor.green if percentage >= 0 else Qt.GlobalColor.red
            items=[
                QTableWidgetItem(metric.capitalize()),
                QTableWidgetItem(f"{value:.2f}"),
                QTableWidgetItem(f"{state_mean:.2f}"),
                QTableWidgetItem(f"{global_mean:.2f}"),
                QTableWidgetItem(f"{sign}{percentage:.1f}%")
            ]
            for col,item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col==4:
                    item.setForeground(color)
                self.metrics_table.setItem(row,col,item)
    
    def computeMeans(self, metric_names: list[str])->tuple[dict,dict]:
        state_stats=self.session.domain_stats.get(self.active_state, {})
        all_stats=self.session.domain_stats
        state_means={}
        global_means={}

        for metric in metric_names:
            state_values=[dictionary[metric] for dictionary in state_stats.values()
                          if metric in dictionary and dictionary["area"]>=self.min_area]
            state_means[metric]=sum(state_values)/len(state_values) if state_values else 0

            all_values=[
                dictionary[metric] for stat in all_stats.values()
                for dictionary in stat.values()
                if metric in dictionary and dictionary["area"]>=self.min_area
            ]
            global_means[metric]=sum(all_values)/len(all_values) if all_values else 0

        return state_means,global_means

    
