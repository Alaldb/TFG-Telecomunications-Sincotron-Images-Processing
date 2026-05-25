# Paleta de colores centralizada
COLORS = {
    "bg":           "#F0F2F5",
    "panel":        "#FFFFFF",
    "border":       "#D0D7DE",
    "accent":       "#2E6DA4",
    "accent_hover": "#1A4F7A",
    "cancel":       "#6C757D",
    "text":         "#1C2526",
    "text_secondary": "#5A6472",
    "vlow":         "#E05C2A",
    "vhigh":        "#2AB5A0",
}

def app_stylesheet() -> str:
    colors = COLORS
    return f"""
        QMainWindow, QWidget {{
            background-color: {colors['bg']};
            color: {colors['text']};
            font-family: 'Segoe UI', sans-serif;
            font-size: 13px;
        }}
        QLabel {{
            color: {colors['text']};
        }}
        QPushButton {{
            background-color: {colors['accent']};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            font-size: 13px;
        }}
        QPushButton:hover {{
            background-color: {colors['accent_hover']};
        }}
        QPushButton#cancel_btn {{
            background-color: {colors['cancel']};
        }}
        QPushButton#cancel_btn:hover {{
            background-color: #5a6268;
        }}
        QLineEdit {{
            border: 1px solid {colors['border']};
            border-radius: 4px;
            padding: 6px 10px;
            background-color: {colors['panel']};
            color: {colors['text']};
        }}
        QLineEdit:focus {{
            border: 1px solid {colors['accent']};
        }}
        QSlider::groove:horizontal {{
            height: 4px;
            background: {colors['border']};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {colors['accent']};
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
    """