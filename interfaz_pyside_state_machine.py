from enum import Enum, auto
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QRadioButton, QButtonGroup, QLineEdit, QApplication,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QWidget,
)

class Screen(Enum):
        PATH_SCREEN = auto()
        METHOD_SCREEN = auto()

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.state = Screen.PATH_SCREEN
        self.placeholder_inputs = {}
        self.create_main_ui()

    def set_state(self, new_state):
        self.state = new_state

    def create_main_ui(self):
        #Configuración del window en si
        self.setWindowTitle("Image Processor")
        self.resize(700, 500)
        self.setMinimumSize(700, 500)
        self.setMaximumSize(700, 500)
        self.setStyleSheet("background-color: #fbf7f2;")

        #Creación del container que ocupa toda la ventana
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(8)
        self.centralWidget.setLayout(self.layout)

        #Creacion del container donde va el título de la app
        self.create_title_container()

        #Creación del container donde van las instrucciones de la ventana
        self.create_subtitle_container()

        #Creación del container donde van los parámetros de entrada
        self.create_params_container()

        #Buttons
        self.create_buttons()

    def create_title_container(self):
        containerTitle = QWidget()
        containerTitle.setFixedHeight(80)
        containerTitle.setStyleSheet("""
            background-color: #f2e3e6;
            border-radius: 10px;
        """)

        titleLayout = QHBoxLayout()
        containerTitle.setLayout(titleLayout)

        title = QLabel("IMAGE PROCESSOR")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            color: #2f2a28;
            letter-spacing: 2px;
        """)

        titleLayout.addWidget(title)
        self.layout.addWidget(containerTitle)
    
    def create_subtitle_container(self):
        containerSubtitle = QWidget()
        containerSubtitle.setFixedHeight(40)

        subtitleLayout = QHBoxLayout()
        containerSubtitle.setLayout(subtitleLayout)

        subtitle = QLabel("Configure input parameters")
        subtitle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        subtitle.setAlignment(Qt.AlignLeft)
        subtitle.setStyleSheet("""
            font-size: 16px;
            color: #3d3633;
            font-weight: 500;
        """)

        subtitleLayout.addWidget(subtitle)
        self.layout.addWidget(containerSubtitle)

    def create_params_container(self):
        self.containerParams = QWidget()
        self.containerParamsLayout = QVBoxLayout()
        self.containerParamsLayout.setAlignment(Qt.AlignTop)
        self.containerParams.setLayout(self.containerParamsLayout)
        if self.state == Screen.PATH_SCREEN:
            self.create_ui_for_path_screen()
        else:
            pass
        self.layout.addWidget(self.containerParams)

    def create_ui_for_path_screen(self):
        self.create_path_container()
        self.create_method_container()
    
    def create_ui_for_method_screen(self):
        if self.ising.isChecked():
            self.create_ising_containers()
        else:
            self.create_thresholding_containers()

    def create_path_container(self):
        pathContainer = QWidget()
        pathLayout = QHBoxLayout()
        pathContainer.setLayout(pathLayout)

        pathLabel = QLabel("Image / Directory:")
        pathLabel.setStyleSheet("color: #4a403b;")

        self.pathInput = QLineEdit()
        self.pathInput.setPlaceholderText(r"C:\path\to\image")
        self.pathInput.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #e0d6cf;
            border-radius: 6px;
            padding: 5px;
        """)
        self.placeholder_inputs["path"] = self.pathInput # Guardamos referencia al input para limpiar su contenido después

        pathLayout.addWidget(pathLabel)
        pathLayout.addWidget(self.pathInput)
        self.containerParamsLayout.addWidget(pathContainer)
    
    def create_method_container(self):
        methodContainer = QWidget()
        methodLayout = QHBoxLayout()
        methodContainer.setLayout(methodLayout)

        methodLabel = QLabel("Processing Method:")
        methodLabel.setStyleSheet("color: #4a403b;")

        self.ising = QRadioButton("Ising")
        self.thresholding = QRadioButton("Thresholding")

        self.group = QButtonGroup()
        self.group.addButton(self.ising)
        self.group.addButton(self.thresholding)

        self.ising.setChecked(True)

        radio_style = """
            QRadioButton {
                color: #3a332f;
                spacing: 6px;
            }

            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 1px solid #3a332f;
                background: white;
            }

            QRadioButton::indicator:checked {
                background-color: black;
                border: 1px solid black;
            }
        """

        self.ising.setStyleSheet(radio_style)
        self.thresholding.setStyleSheet(radio_style)

        methodLayout.addWidget(methodLabel)
        methodLayout.addWidget(self.ising)
        methodLayout.addWidget(self.thresholding)

        self.containerParamsLayout.addWidget(methodContainer)

    def create_ising_containers(self):
        title = QLabel("Ising parameters")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        num_states_description = "Number of possible states in the image\n- Has to be positive\n- Has to be greater than 1\n- Has to be integer [0, 1, 2, ...]"
        beta_description = "Inverse temperature parameter controlling interaction strength\n- Must be >= 0\n- 0 gives random behavior\n- High values may oversmooth and trap in local minima"
        max_iter_description = "Max iterations (int)\n- Must be positive integer\n- Higher values may increase processing time but can lead to better convergence"

        param_names=["num_states", "beta", "max_iterations"]
        placeholders=["Default: 3", "Default: 2.0", "Default: 10"]
        descriptions=[num_states_description, beta_description, max_iter_description]

        for parameter, placeholder, description in zip(param_names, placeholders, descriptions):
            self.build_parameters_container(
                self.containerParamsLayout,
                parameter,
                placeholder,
                description
            )

    def create_thresholding_containers(self):
        title = QLabel("Thresholding parameters")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        clip_description = "Contrast limit for adaptive histogram equalization\n- Must be > 0\n- Higher values increase contrast\n- Too high may amplify noise"
        grid_description = "Grid size for local histogram equalization\n- Must be positive integers (int, int)\n- Defines number of tiles in x and y\n- Smaller grid enhances local contrast"
        ksize_description = "Kernel size for Gaussian blur\n- Must be positive odd integers (int, int)\n- Example: (3,3), (5,5)\n- Larger values produce stronger smoothing"
        sigma_description = "Standard deviation for Gaussian kernel\n- Must be >= 0\n- 0 lets OpenCV compute it automatically\n- Higher values increase smoothing"

        param_names = ["clip_limit", "grid_size", "ksize", "sigma"]
        placeholders = ["Default: 2.0", "Default: (8,8)", "Default: (5,5)", "Default: 0"]
        descriptions = [clip_description, grid_description, ksize_description, sigma_description]

        for parameter, placeholder, description in zip(param_names, placeholders, descriptions):
            self.build_parameters_container(
                self.containerParamsLayout,
                parameter,
                placeholder,
                description
            )

    def create_right_button(self):
        rightButton = QPushButton()
        rightButton.setStyleSheet("""
                QPushButton {
                    background-color: #5a4336;
                    color: white;
                    padding: 8px 14px;
                    border-radius: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #7a5a48;
                }
                QPushButton:pressed {
                    background-color: #3e2e25;
                }
            """)
        if self.state == Screen.PATH_SCREEN:
            rightButton.setText("NEXT →")
            rightButton.clicked.connect(lambda: self.change_screen(Screen.METHOD_SCREEN))
        elif self.state == Screen.METHOD_SCREEN:
            rightButton.setText("EXECUTE")
            rightButton.clicked.connect(lambda: self.execute_processing())
        else:
            pass
        return rightButton

    def create_left_button(self):
        if self.state == Screen.METHOD_SCREEN:
            leftButton=QPushButton("← BACK")
            leftButton.setStyleSheet(leftButton.styleSheet())
            leftButton.setStyleSheet(
                            """
                    QPushButton {
                        background-color: #cccccc;
                        color: black;
                        padding: 8px 14px;
                        border-radius: 8px;
                    }
                """
                )
            leftButton.clicked.connect(lambda: self.change_screen(Screen.PATH_SCREEN))
            return leftButton
        return None
        
    def create_buttons(self):
        if not hasattr(self, "buttonLayout"):
            self.buttonContainer = QWidget()
            self.buttonLayout = QHBoxLayout()
            self.buttonContainer.setLayout(self.buttonLayout)
            self.layout.addWidget(self.buttonContainer)

        self.clean_container(self.buttonLayout)

        right_button = self.create_right_button()
        left_button = self.create_left_button()

        self.buttonLayout.addStretch()

        if left_button:
            self.buttonLayout.addWidget(left_button)

        self.buttonLayout.addWidget(right_button)

    def clean_container(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def change_screen(self, new_screen):
        self.set_state(new_screen)
        self.clean_container(self.containerParamsLayout)
        self.clean_container(self.buttonLayout)
        if new_screen == Screen.PATH_SCREEN:
            self.create_ui_for_path_screen()
            self.create_buttons()
        elif new_screen == Screen.METHOD_SCREEN:
            self.create_ui_for_method_screen()
            self.create_buttons()
        else:
            pass

    def build_parameters_container(self, container_layout, param_name, placeholder_content, description):
        container = QWidget()
        container_layout_h = QHBoxLayout()
        container.setLayout(container_layout_h)

        # --- PARAM NAME ---
        title = QLabel(param_name + ":")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setFixedWidth(120)  # <-- todos empiezan en la misma columna

        # --- INPUT ---
        input_field = QLineEdit()
        input_field.setPlaceholderText(placeholder_content)
        self.placeholder_inputs[param_name] = input_field # Guardamos referencia al input para limpiar su contenido después

        input_field.setFixedWidth(80)  # suficiente para "[1,9]"
        input_field.setAlignment(Qt.AlignRight)  # <-- contenido a la derecha

        # --- DESCRIPTION ---
        description_label = QLabel(description)
        description_label.setWordWrap(True)

        description_label.setStyleSheet("""
            color: #4a403b;
            font-size: 12px;
        """)

        # Justificado (requiere rich text)
        description_html = description.replace("\n", "<br>")
        description_label.setText(
            f"<div style='text-align: justify;'>{description_html}</div>"
        )

        # --- LAYOUT ---
        container_layout_h.addWidget(title)
        container_layout_h.addWidget(input_field)
        container_layout_h.addWidget(description_label, stretch=1)

        container_layout.addWidget(container)

    def save_method_in_parameters_path_screen(self):
        parameters={}
        parameters["method"]="ising" if self.ising.isChecked() else "thresholding"
        parameters["path"]=self.placeholder_inputs["path"].text() if self.pathInput.text() else None
        self.placeholder_inputs["path"].remove()
    def save_inputs_in_parameters_method_screen(self):
        parameters={}
        
        for param_name, input in self.placeholder_inputs.items():
            value = input.text() if input.text() else None
            if value is not None:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            parameters[param_name] = value
        return parameters
    
    def execute_processing(self):
        parameters = self.save_inputs_in_parameters()
        print("Executing with parameters:", parameters)

        
    


app = QApplication([])
window = MainWindow()
window.show()
app.exec()