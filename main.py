import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from interface.mainWindow import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
window.showMaximized()
sys.exit(app.exec())