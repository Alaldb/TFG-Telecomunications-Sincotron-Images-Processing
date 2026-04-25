import unittest
from interfaz_pyside_state_machine import MainWindow, Screen
from enum import Enum, auto
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QRadioButton, QButtonGroup, QLineEdit, QApplication,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QWidget,
)

class TestMainWindow(unittest.TestCase):

    def test_unittest_funcionality(self):
        self.assertEqual(1,1)

    def test_window_creation(self):
        window = MainWindow()
        self.assertIsNotNone(window)
    
    def test_state_transition(self):
        window = MainWindow()
        # estado inicial
        window.set_state(Screen.PATH_SCREEN)
        self.assertEqual(window.state, Screen.PATH_SCREEN)
    
    def test_main_ui_creation(self):
        window = MainWindow()
        self.assertIsNotNone(window)
        self.assertEqual(window.windowTitle(), "Image Processor")
        self.assertEqual(window.minimumWidth(), 700)
        self.assertEqual(window.minimumHeight(), 500)
        self.assertIsNotNone(window.centralWidget)
        self.assertIsNotNone(window.layout)
        self.assertGreater(window.layout.count(), 0)
    
    def test_create_params_container(self):
        window = MainWindow()
        window.create_params_container()
        self.assertIsNotNone(window.containerParams)
        self.assertIsNotNone(window.containerParamsLayout)
        self.assertEqual(window.containerParams.layout(), window.containerParamsLayout)
        self.assertEqual(
            window.containerParamsLayout.alignment(),
            Qt.AlignTop
        )
    def test_create_title_container(self):
        window = MainWindow()
        initial_count = window.layout.count()

        window.create_title_container()

        self.assertEqual(window.layout.count(), initial_count + 1)

        item = window.layout.itemAt(window.layout.count() - 1)
        self.assertIsInstance(item.widget(), QWidget)


    def test_create_subtitle_container(self):
        window = MainWindow()
        initial_count = window.layout.count()

        window.create_subtitle_container()

        self.assertEqual(window.layout.count(), initial_count + 1)

        item = window.layout.itemAt(window.layout.count() - 1)
        self.assertIsInstance(item.widget(), QWidget)


    def test_create_ui_for_path_screen(self):
        window = MainWindow()

        # limpiar para aislar test
        window.containerParamsLayout = QVBoxLayout()
        window.containerParams = QWidget()
        window.containerParams.setLayout(window.containerParamsLayout)

        window.create_ui_for_path_screen()

        # debería haber 2 widgets: path + method
        self.assertEqual(window.containerParamsLayout.count(), 2)


    def test_create_path_container(self):
        window = MainWindow()

        window.containerParamsLayout = QVBoxLayout()
        window.containerParams = QWidget()
        window.containerParams.setLayout(window.containerParamsLayout)

        window.create_path_container()

        self.assertEqual(window.containerParamsLayout.count(), 1)

        # comprobar que existe el input
        self.assertIsInstance(window.pathInput, QLineEdit)
        self.assertEqual(window.pathInput.placeholderText(), r"C:\path\to\image")


    def test_create_method_container(self):
        window = MainWindow()

        window.containerParamsLayout = QVBoxLayout()
        window.containerParams = QWidget()
        window.containerParams.setLayout(window.containerParamsLayout)

        window.create_method_container()

        self.assertEqual(window.containerParamsLayout.count(), 1)

        # comprobar radio buttons
        self.assertIsInstance(window.ising, QRadioButton)
        self.assertIsInstance(window.thresholding, QRadioButton)

        # comprobar grupo
        self.assertIsInstance(window.group, QButtonGroup)

        # comprobar default checked
        self.assertTrue(window.ising.isChecked())
        self.assertFalse(window.thresholding.isChecked())

        def test_create_buttons(self):
            window = MainWindow()

            # asegurar layout limpio
            window.buttonLayout = QHBoxLayout()
            window.buttonContainer = QWidget()
            window.buttonContainer.setLayout(window.buttonLayout)

            initial_count = window.buttonLayout.count()

            window.create_right_button()

            self.assertEqual(window.buttonLayout.count(), initial_count + 1)
            self.assertIsInstance(window.rightButton, QPushButton)

    def test_create_right_button_path_screen(self):
        window = MainWindow()
        window.state = Screen.PATH_SCREEN

        btn = window.create_right_button()

        self.assertIsInstance(btn, QPushButton)
        self.assertEqual(btn.text(), "NEXT →")

    def test_create_right_button_method_screen(self):
        window = MainWindow()
        window.state = Screen.METHOD_SCREEN

        btn = window.create_right_button()

        self.assertIsInstance(btn, QPushButton)
        self.assertEqual(btn.text(), "EXECUTE")

    def test_create_left_button_method_screen(self):
        window = MainWindow()
        window.state = Screen.METHOD_SCREEN

        btn = window.create_left_button()

        self.assertIsInstance(btn, QPushButton)
        self.assertEqual(btn.text(), "← BACK")

    def test_create_left_button_path_screen(self):
        window = MainWindow()
        window.state = Screen.PATH_SCREEN

        btn = window.create_left_button()

        self.assertIsNone(btn)
    
    def test_change_screen_to_method(self):
        window = MainWindow()

        window.change_screen(Screen.METHOD_SCREEN)

        self.assertEqual(window.state, Screen.METHOD_SCREEN)
        self.assertGreater(window.containerParamsLayout.count(), 0)
        self.assertGreater(window.buttonLayout.count(), 0)


    def test_change_screen_to_path(self):
        window = MainWindow()

        window.change_screen(Screen.METHOD_SCREEN)
        window.change_screen(Screen.PATH_SCREEN)

        self.assertEqual(window.state, Screen.PATH_SCREEN)
        self.assertGreater(window.containerParamsLayout.count(), 0)


    def test_build_parameters_container(self):
        window = MainWindow()

        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)

        window.build_parameters_container(
            layout,
            "test_param",
            "default",
            "this is a test description"
        )

        self.assertEqual(layout.count(), 1)

        item = layout.itemAt(0)
        self.assertIsInstance(item.widget(), QWidget)


    def test_create_ui_for_method_screen_ising(self):
        window = MainWindow()

        window.ising = QRadioButton()
        window.thresholding = QRadioButton()
        window.ising.setChecked(True)

        window.containerParamsLayout = QVBoxLayout()
        window.containerParams = QWidget()
        window.containerParams.setLayout(window.containerParamsLayout)

        window.create_ui_for_method_screen()

        self.assertGreater(window.containerParamsLayout.count(), 0)


    def test_create_ui_for_method_screen_thresholding(self):
        window = MainWindow()

        window.ising = QRadioButton()
        window.thresholding = QRadioButton()
        window.thresholding.setChecked(True)

        window.containerParamsLayout = QVBoxLayout()
        window.containerParams = QWidget()
        window.containerParams.setLayout(window.containerParamsLayout)

        window.create_ui_for_method_screen()

        self.assertGreater(window.containerParamsLayout.count(), 0)


    def test_clean_container(self):
        window = MainWindow()

        layout = QVBoxLayout()
        widget = QWidget()
        layout.addWidget(widget)

        self.assertEqual(layout.count(), 1)

        window.clean_container(layout)

        self.assertEqual(layout.count(), 0)


    def test_buttons_rebuild_on_change_screen(self):
        window = MainWindow()

        initial = window.buttonLayout.count()

        window.change_screen(Screen.METHOD_SCREEN)

        self.assertGreaterEqual(window.buttonLayout.count(), initial)


    def test_no_left_button_in_path_screen_after_change(self):
        window = MainWindow()

        window.change_screen(Screen.METHOD_SCREEN)
        window.change_screen(Screen.PATH_SCREEN)

        # en path screen no debería existir leftButton
        self.assertFalse(hasattr(window, "leftButton"))


if __name__ == "__main__":
    unittest.main(verbosity=2)