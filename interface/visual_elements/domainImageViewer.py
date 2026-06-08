from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QMouseEvent

from interface.styles import COLORS

class DomainImageViewer(QLabel):

    domain_clicked=Signal(int,dict)

    def __init__(self, parent=None)->None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200,200)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.rgb_image: np.ndarray|None=None
        self.labeled_image: np.ndarray|None=None
        self.domain_stats: dict={}
        self.selected_domain: int|None=None

        self.scale: float=1
        self.offset_x: float=0
        self.offset_y: float=0
    
    def setData(self, rgb_image: np.ndarray, labeled_image: np.ndarray, domain_stats: dict)->None:
        self.rgb_image=rgb_image
        self.labeled_image=labeled_image
        self.domain_stats=domain_stats
        self.selected_domain=None
        self.renderImage()
    
    def renderImage(self)->None:
        if self.rgb_image is None:
            return
        
        display_image=self.rgb_image.copy()

        if self.selected_domain is not None:
            mask=self.labeled_image!=self.selected_domain
            display_image[mask]=(display_image[mask]*0.25).astype(np.uint8)

        display_height,display_width,_=display_image.shape
        display_image=np.ascontiguousarray(display_image)
        q_img=QImage(display_image.data,display_width,display_height,display_width*3,QImage.Format.Format_RGB888)

        label_width=self.width()
        label_height=self.height()
        self.scale=min(label_width/display_width,label_height/display_height)
        scaled_width=int(display_width*self.scale)
        scaled_height=int(display_height*self.scale)
        self.offset_x=(label_width-scaled_width)/2
        self.offset_y=(label_height-scaled_height)/2

        pixmap=QPixmap.fromImage(q_img).scaled(
            QSize(scaled_width, scaled_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event: QMouseEvent)->None:
        if self.rgb_image is None:
            return
        click_x=event.position().x()
        click_y=event.position().y()
        img_x=int((click_x-self.offset_x)/self.scale)
        img_y=int((click_y-self.offset_y)/self.scale)

        img_height,img_width,_=self.rgb_image.shape
        if img_x<0 or img_y<0 or img_x>=img_width or img_y>=img_height:
            return
        
        domain_id=int(self.labeled_image[img_y,img_x])#remember np indexes y,x in images
        if domain_id==0:
            return
        
        self.selected_domain=domain_id
        self.renderImage()
        self.domain_clicked.emit(domain_id,self.domain_stats.get(domain_id,{}))#this avoids errors id by any  mean the key does not exist, it returns an empty dict

    def resizeEvent(self, event)->None:
        super().resizeEvent(event)
        self.renderImage()
