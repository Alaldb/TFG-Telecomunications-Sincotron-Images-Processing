from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt

from interface.styles import COLORS

class ComparisonResultsWindow(QDialog):
    def __init__(self,result:dict,parent=None)->None:
        super().__init__(parent)
        self.setWindowTitle("Comparison Results")
        self.setMinimumSize(700,400)

        self.result=result
        self.stat_diffs=result["stat_diffs"]
        self.displacement=result["displacement"]

        self.buildUi()
        self.loadFirstMetric()

    def buildUi(self)->None:
        base=QVBoxLayout(self)
        base.setContentsMargins(24,20,24,16)
        base.setSpacing(12)

        base.addLayout(self.buildMetricsSelector())
        base.addWidget(self.buildStatsTable(), stretch=1)
        base.addWidget(self.buildDisplacementPanel())

    def buildMetricsSelector(self)->QHBoxLayout:
        row=QHBoxLayout()
        label=QLabel("Select Metric")
        label.setStyleSheet(f"font-size: 12px; color: {COLORS['text_secondary']};")

        self.metric_combo=QComboBox()
        self.metric_combo.setFixedWidth(200)
        for metric in self.stat_diffs.keys():
            self.metric_combo.addItem(metric.replace("_"," ").capitalize())
        self.metric_combo.currentIndexChanged.connect(self.onMetricChanged)

        row.addWidget(label)
        row.addWidget(self.metric_combo)
        row.addStretch()
        return row
    
    def buildStatsTable(self)->QFrame:
        frame=QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; background-color: {COLORS['panel']}; }}"
        )
        table_layout=QVBoxLayout(frame)
        table_layout.setContentsMargins(12,12,12,12)

        self.stats_table=QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Domain A", "Domain B", "Difference", "%"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.verticalHeader().setVisible(False)

        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stats_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        self.stats_table.setStyleSheet("border: none;")
        table_layout.addWidget(self.stats_table)
        return frame
    
    def buildDisplacementPanel(self)->QFrame:
        frame=QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; background-color: {COLORS['panel']}; }}"
        )
        displacement_layout=QVBoxLayout(frame)
        displacement_layout.setContentsMargins(12,12,12,12)
        displacement_layout.setSpacing(6)

        title=QLabel("Displacement")
        title.setStyleSheet(
            f"font-size: 12px; font-weight: bold; color: {COLORS['text_secondary']}; border: none;"
        )

        rows=[
            ("Centroid A (y, x)", f"{self.displacement['centroid_a'][0]:.3f}, {self.displacement['centroid_a'][1]:.3f}"),
            ("Centroid B (y, x)", f"{self.displacement['centroid_b'][0]:.3f}, {self.displacement['centroid_b'][1]:.3f}"),
            ("Displacement Vector", f"{self.displacement['displacement_vector'][0]:.3f}, {self.displacement['displacement_vector'][1]:.3f}"),
            ("Vector Module", f"{self.displacement['vector_module']:.3f}"),
            ("Normalized Module", f"{self.displacement['normalized_vector']:.3f}"),
        ]

        displacement_layout.addWidget(title)
        for name_text, value_text in rows:
            row=QHBoxLayout()
            name_lbl=QLabel(name_text)
            name_lbl.setStyleSheet(
                f"font-size: 12px; color: {COLORS['text_secondary']}; border: none;"
            )
            value_lbl=QLabel(value_text)
            value_lbl.setStyleSheet(
                f"font-size: 12px; color: {COLORS['text']}; border: none;"
            )
            row.addWidget(name_lbl)
            row.addStretch()
            row.addWidget(value_lbl)
            displacement_layout.addLayout(row)
        return frame
    
    def onMetricChanged(self, index:int)->None:
        metric_key=list(self.stat_diffs.keys())[index]
        self.updateStatTable(metric_key)
    
    def updateStatTable(self, metric_key:str)->None:
        data=self.stat_diffs[metric_key]
        self.stats_table.setRowCount(1)
        percentage=data["percentage"]
        sign="+" if percentage>=0 else ""
        color= Qt.GlobalColor.green if percentage>=0 else Qt.GlobalColor.red

        items=[
            QTableWidgetItem(f"{data['value_a']:.3f}"),
            QTableWidgetItem(f"{data['value_b']:.3f}"),
            QTableWidgetItem(f"{data['absolute']:.3f}"),
            QTableWidgetItem(f"{sign}{percentage:.2f}%"),
        ]

        for column, item in enumerate(items):
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if column==3:
                item.setForeground(color)
            self.stats_table.setItem(0,column,item)

    def loadFirstMetric(self)->None:
        if self.stat_diffs:
            self.metric_combo.setCurrentIndex(0)
            first_key=list(self.stat_diffs.keys())[0]
            self.updateStatTable(first_key)
    


