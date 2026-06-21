# Domain Analysis Tool — Developer Guide

**[Español](#español) | [English](#english)**

---

## Español

## Índice

1. [¿Qué es y qué busca?](#1-qué-es-y-qué-busca)
2. [Arquitectura y flujo de datos](#2-arquitectura-y-flujo-de-datos)
3. [Librerías y su utilidad](#3-librerías-y-su-utilidad)
4. [Organización de directorios](#4-organización-de-directorios)
5. [Cómo añadir funcionalidad nueva](#5-cómo-añadir-funcionalidad-nueva)
6. [Servicios actuales](#6-servicios-actuales)
7. [Paneles actuales](#7-paneles-actuales)
8. [Contenedores de datos y persistencia](#8-contenedores-de-datos-y-persistencia)
9. [Widgets especiales](#9-widgets-especiales)
10. [Inputs y outputs](#10-inputs-y-outputs)

---

## 1. ¿Qué es y qué busca?

Domain Analysis Tool es una aplicación de escritorio para el análisis cuantitativo de imágenes de microscopía científica, orientada al estudio de dominios magnéticos en imágenes XMCD. El objetivo es ofrecer una herramienta que permita segmentar una imagen en estados de intensidad, identificar y etiquetar individualmente cada dominio dentro de cada estado, y calcular métricas geométricas por dominio.

**Principios de diseño:**

- **Modularidad** — cada responsabilidad está aislada en su propio módulo. Los servicios de procesamiento no conocen la interfaz; los paneles no conocen los algoritmos.
- **Extensibilidad** — añadir un nuevo método de segmentación, una nueva métrica o un nuevo panel no debe requerir tocar código existente más allá del punto de integración.
- **Separación de capas** — procesamiento, interfaz, persistencia y estadísticas son capas independientes que se comunican a través de contenedores de datos bien definidos (`Session`, `SegmentationContainer`).

---

## 2. Arquitectura y flujo de datos

```
Imagen .tif
    │
    ▼
Corrector                  ← processing/corrector.py
    │  (corrected_image: np.ndarray)
    ▼
Segmentación
    ├── Ising (ICM)         ← processing/isingMethodService.py
    └── GraphCutsService    ← processing/graphCutsService.py
    │  (SegmentationContainer)
    ▼
DomainService              ← processing/domainService.py
    │  (labeled_images, colored_images, domain_data)
    ▼
computeDomainStats         ← stats/domainStats.py
    │  (domain_stats: {estado: {domain_id: {métrica: valor}}})
    ▼
Session                    ← core/session.py
    │  (contenedor con toda la información del análisis)
    ▼
ResultsPanel               ← interface/panels/resultsPanel.py
    ├── DomainExplorerWindow
    └── Exportación (.session / .tif / .xlsx)
```

`Session` es el objeto central. Se construye en `MainWindow` una vez finalizada la segmentación y se pasa entre paneles. Todo lo que un panel necesita está dentro de `Session`; los paneles no llaman directamente a los servicios de procesamiento (excepto `ResultsPanel`, que lanza `DomainsWorker` en background si la sesión viene de carga directa sin `domain_data` calculado).

---

## 3. Librerías y su utilidad

| Librería | Uso en el proyecto |
|---|---|
| **PySide6** | Framework de interfaz gráfica. Todos los paneles, ventanas y widgets heredan de clases Qt. |
| **numpy** | Operaciones sobre arrays de imágenes (matrices 2D de píxeles, máscaras, imágenes RGB). Es la base de todo el procesamiento. |
| **opencv-python** | Lectura de imágenes `.tif`, normalización, detección de círculos (Hough), desenfoque gaussiano para corrección de iluminación. |
| **scikit-image** | `regionprops` para calcular métricas geométricas por dominio; `label` para etiquetar regiones conectadas. |
| **scikit-learn** | KMeans para la inicialización del modelo Ising; GaussianMixture (GMM) para el término de datos en Graph Cuts. |
| **scipy** | `gaussian_kde` para la curva de densidad en los histogramas de segmentación. |
| **matplotlib** | Renderizado de histogramas embebidos en los paneles (via `FigureCanvasQTAgg`). |
| **openpyxl** | Escritura de archivos `.xlsx` para la exportación de métricas. |
| **seaborn** | Visualizaciones estadísticas auxiliares (usado en exploración, no en producción de la interfaz). |
| **tifffile** | Lectura de imágenes TIFF con mayor fidelidad que OpenCV para formatos científicos multi-canal o de alta profundidad. |
| **gco-wrapper** | Implementación eficiente de Graph Cuts (alpha-expansion) para la segmentación por cortes de grafo. |

---

## 4. Organización de directorios

```
New_Tool/
│
├── main.py                        # Punto de entrada
│
├── core/                          # Estructuras de datos centrales
│   ├── session.py                 # Dataclass Session
│   ├── segmentationContainer.py   # Dataclass SegmentationContainer + Enum SegmentationMethod
│   └── pipeline.py                # PipelineDictator: orquestador headless del flujo completo
│
├── processing/                    # Servicios de procesamiento (sin dependencias de UI)
│   ├── corrector.py               # Corrección de iluminación, estiramiento, brillo/contraste
│   ├── isingMethodService.py      # Segmentación ICM (modelo de Ising)
│   ├── graphCutsService.py        # Segmentación por Graph Cuts
│   └── domainService.py           # Etiquetado de dominios y extracción de datos por dominio
│
├── stats/                         # Cálculo de métricas
│   └── domainStats.py             # computeDomainStats: área, perímetro, roughness
│
├── persistence/                   # Lectura y escritura de archivos
│   └── session_io.py              # saveSession, loadSession, exportCorrectedImage, exportDataExcel
│
├── interface/                     # Capa de interfaz gráfica
│   ├── styles.py                  # Paleta de colores y stylesheet global
│   ├── mainWindow.py              # QMainWindow: orquestador de paneles
│   │
│   ├── panels/                    # Un archivo por pantalla/panel
│   │   ├── startPanel.py
│   │   ├── loadPanel.py
│   │   ├── bcPanel.py
│   │   ├── icmPanel.py
│   │   ├── graphCutsPanel.py
│   │   ├── resultsPanel.py
│   │   ├── domainExplorerWindow.py
│   │   └── domainComparisonPanel.py
│   │
│   └── visual_elements/           # Widgets reutilizables con lógica propia
│       └── domainImageViewer.py
│
└── Elements/                      # Recursos externos (rutas de fiji, imágenes temporales)
```

**Lógica de separación:**
- `processing/` nunca importa de `interface/`. Los servicios reciben y devuelven datos puros (numpy arrays, dataclasses).
- `interface/` nunca implementa lógica de procesamiento. Los paneles llaman a servicios y muestran resultados.
- `core/` define las estructuras compartidas entre capas.
- `stats/` es independiente: recibe arrays etiquetados y devuelve diccionarios de métricas.
- `persistence/` solo lee y escribe. No transforma datos.

---

## 5. Cómo añadir funcionalidad nueva

### Estructura general de un servicio

Un servicio en `processing/` es una clase que recibe datos en el constructor o en un método `run()` y expone los resultados mediante `get_data()` o un método `getContainer()`.

```python
# processing/miServicio.py
import numpy as np
from core.segmentationContainer import SegmentationContainer

class MiServicio:
    def __init__(self, param1: float, param2: int):
        self.param1 = param1
        self.param2 = param2

    def run(self, image: np.ndarray) -> None:
        # procesamiento
        self.result = ...

    def getSegmentationContainer(self) -> SegmentationContainer:
        return SegmentationContainer(
            original_image=...,
            mask=...,
            final_image=self.result,
            num_states=...,
            parameters=...,
            initial_labels=...,
            method=SegmentationMethod.MI_METODO,
        )
```

Si el método es un nuevo tipo de segmentación, añade su entrada al enum `SegmentationMethod` en `core/segmentationContainer.py`:

```python
class SegmentationMethod(Enum):
    ICM = "ICM"
    GRAPH_CUTS = "GraphCuts"
    MI_METODO = "MiMetodo"   # nuevo
```

### Estructura general de un panel

Un panel es un `QWidget` que:
- Emite señales para comunicarse con `MainWindow` (nunca llama directamente a otros paneles).
- Contiene un worker `QThread` si lanza procesamiento pesado en background.
- Separa la construcción de la UI (`buildUi`) de la lógica de datos (`loadData`, `onWorkerFinished`).

```python
# interface/panels/miPanel.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal

class MiPanel(QWidget):
    resultado_aceptado = Signal(object)   # emite el resultado hacia MainWindow
    cancelled = Signal()
    home = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.buildUi()

    def buildUi(self):
        layout = QVBoxLayout(self)
        self.run_but = QPushButton("Run")
        self.run_but.clicked.connect(self.onRunClicked)
        layout.addWidget(self.run_but)

    def loadData(self, datos):
        # recibe datos desde MainWindow y prepara el panel
        self.datos = datos

    def onRunClicked(self):
        # lanza procesamiento y emite señal con el resultado
        resultado = ...
        self.resultado_aceptado.emit(resultado)
```

Si el procesamiento es costoso, usa un worker:

```python
from PySide6.QtCore import QThread, Signal

class MiWorker(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, datos):
        super().__init__()
        self.datos = datos

    def run(self):
        try:
            resultado = MiServicio().run(self.datos)
            self.finished.emit(resultado)
        except Exception as e:
            self.error.emit(str(e))
```

### Integrar el nuevo panel en MainWindow

`MainWindow` usa un `QStackedWidget` para gestionar las pantallas. Para añadir un panel nuevo:

```python
# interface/mainWindow.py

# 1. Importar
from interface.panels.miPanel import MiPanel

# 2. En buildUi(), instanciar y conectar señales
self.mi_panel = MiPanel()
self.mi_panel.resultado_aceptado.connect(self.onMiResultado)
self.mi_panel.cancelled.connect(self.goToStart)
self.mi_panel.home.connect(self.goToStart)
self.stack.addWidget(self.mi_panel)

# 3. Métodos de navegación
def goToMiPanel(self):
    self.mi_panel.loadData(self.datos_necesarios)
    self.stack.setCurrentWidget(self.mi_panel)

def onMiResultado(self, resultado):
    # guardar resultado en self.session y navegar al siguiente paso
    self.session.mi_dato = resultado
    self.goToResultados()
```

---

## 6. Servicios actuales

### `Corrector` — `processing/corrector.py`

Corrección de imagen en cuatro pasos encadenados:

1. **`create_mask`** — detecta el área de interés circular mediante `HoughCircles`. Si no detecta círculo, la máscara cubre toda la imagen.
2. **`correct_illumination`** — corrige gradientes de iluminación restando un fondo estimado con blur gaussiano.
3. **`linear_stretch`** — estira el histograma al rango [0, 255] entre `v_low` y `v_high`.
4. **`adjust_brightness_contrast`** — aplica desplazamiento de brillo y factor de contraste.

El método principal es `apply_correction`, que encadena todos los pasos. Los pasos individuales son públicos y pueden llamarse por separado.

---

### `Ising` — `processing/isingMethodService.py`

Segmentación bayesiana mediante el modelo de Ising con optimización ICM (Iterated Conditional Modes):

1. **Inicialización** con KMeans (`n_clusters = num_states`) sobre los píxeles dentro de la máscara.
2. **ICM** — en cada iteración minimiza la energía por píxel: `E = energía_estadística + energía_vecindad`. La energía estadística mide la distancia del píxel a la media de cada estado (distribución gaussiana); la energía de vecindad favorece que píxeles adyacentes compartan estado (controlada por `beta`).
3. **Convergencia** — el algoritmo para cuando ningún píxel cambia de estado o se alcanza `max_iterations`.

Expone `getSegmentationContainer()` para construir el contenedor estándar.

---

### `GraphCutsService` — `processing/graphCutsService.py`

Segmentación por cortes de grafo con alpha-expansion:

1. **GMM** — modela la distribución de intensidad de cada estado con una mezcla de gaussianas (`GaussianMixture`, `number_gaussians_per_state` componentes).
2. **Término de datos** — probabilidad negativa log de cada píxel bajo cada estado según el GMM.
3. **Término de suavizado (N-link)** — penaliza píxeles adyacentes con estados diferentes, ponderado por `lambda` y la diferencia de intensidad (controlada por `sigma`).
4. **Alpha-expansion** — minimización global de la energía mediante `gco-wrapper`.

Expone `getSegmentationContainer()` con el mismo contrato que `Ising`.

---

### `DomainService` — `processing/domainService.py`

A partir de un `SegmentationContainer`, extrae los dominios individuales:

1. **`extract_state_images`** — genera una imagen binaria por estado (1 dentro del estado y dentro de la máscara, 0 fuera).
2. **`label_domains`** — aplica `skimage.measure.label` con conectividad 8 para etiquetar regiones conexas. Cada dominio recibe un id único por estado.
3. **`color_domains`** — asigna un color aleatorio a cada dominio para visualización.
4. **`extract_domain_data`** — extrae coordenadas y valores de intensidad por dominio.

El método `get_data()` devuelve un diccionario con todas estas estructuras.

---

### `computeDomainStats` — `stats/domainStats.py`

Calcula métricas geométricas para cada dominio usando `skimage.measure.regionprops`. Itera sobre las imágenes etiquetadas y construye la estructura:

```
{ estado: { domain_id: { "area": float, "perimeter": float, "roughness": float } } }
```

Las funciones de cada métrica son independientes y reciben el objeto `domain` de `regionprops`. Ver sección [Cómo añadir una nueva métrica](#cómo-añadir-una-nueva-métrica-en-domainstatspy) en el README de usuario.

---

## 7. Paneles actuales

### `StartPanel` — `interface/panels/startPanel.py`

Pantalla de inicio. Emite `analyse_requested` para ir al flujo de análisis o `compare_requested` para ir a la comparación de sesiones.

---

### `LoadPanel` — `interface/panels/loadPanel.py`

Carga una imagen `.tif` con OpenCV o una sesión `.session` mediante `loadSession`. Permite seleccionar el método de segmentación (ICM / Graph Cuts) antes de continuar. Emite `image_loaded(np.ndarray, str)` o `session_loaded(Session)`.

---

### `BcPanel` — `interface/panels/bcPanel.py`

Corrección de brillo y contraste con previsualización en tiempo real. Llama a `Corrector` directamente sin worker (la corrección es rápida). Emite `correction_accepted(np.ndarray, int, int)` con la imagen corregida y los valores de v_low / v_high.

---

### `IsingPanel` — `interface/panels/icmPanel.py`

Segmentación ICM. Lanza `IsingWorker` (QThread) con los parámetros configurados. Muestra la imagen original, la vista por estado seleccionado, el histograma de intensidades y la leyenda de distribución por estado. Emite `segmentation_accepted(SegmentationContainer)`.

---

### `GraphCutsPanel` — `interface/panels/graphCutsPanel.py`

Segmentación por Graph Cuts. Misma estructura visual que `IsingPanel` con parámetros propios (Lambda, Sigma, Gaussians, Iterations). Lanza `GraphCutsWorker`. Emite `segmentation_accepted(SegmentationContainer)`.

---

### `ResultsPanel` — `interface/panels/resultsPanel.py`

Panel principal de resultados. Recibe una `Session` y muestra:
- Imagen original y vista de dominios coloreados.
- Histograma de la métrica activa por estado.
- Pestañas de estado y botones de métrica.
- Controles de filtro de área y rango de histograma.

Si la sesión tiene `segmentation_container` pero no `domain_stats`, lanza `DomainsWorker` en background para calcularlos. Desde aquí se abre `DomainExplorerWindow` y se accede a las exportaciones.

---

### `DomainExplorerWindow` — `interface/panels/domainExplorerWindow.py`

Ventana auxiliar (`QDialog`) que permite inspeccionar dominios individualmente. Usa `DomainImageViewer` para la imagen interactiva y una tabla (`QTableWidget`) para mostrar las métricas del dominio seleccionado junto a la media del estado y la media global, con el porcentaje de desviación.

---

### `DomainComparisonPanel` — `interface/panels/domainComparisonPanel.py`

Permite cargar dos sesiones `.session` y comparar sus métricas lado a lado.

---

## 8. Contenedores de datos y persistencia

### `SegmentationContainer` — `core/segmentationContainer.py`

Dataclass que encapsula el resultado de cualquier método de segmentación:

```python
@dataclass
class SegmentationContainer:
    original_image: np.ndarray      # imagen de entrada normalizada
    mask: np.ndarray                # máscara booleana del área de interés
    final_image: np.ndarray         # imagen segmentada (valores 0..num_states-1)
    num_states: int                 # número de estados
    parameters: dict                # {estado: {"mean": float, "std": float}}
    initial_labels: np.ndarray      # etiquetas iniciales (antes de optimización)
    method: SegmentationMethod      # ICM o GRAPH_CUTS
    method_configuration: dict      # parámetros usados (beta, lambda, etc.)
```

Es el contrato entre los servicios de segmentación y el resto del sistema. `DomainService` lo consume directamente.

---

### `Session` — `core/session.py`

Dataclass central que acumula todo el estado del análisis:

```python
@dataclass
class Session:
    image_name: str
    original_image: np.ndarray | None
    corrected_image: np.ndarray | None
    ising_result: np.ndarray | None
    domain_data: dict                   # salida de DomainService.get_data()
    parameters: dict                    # configuración usada
    ising_stats: dict                   # parámetros estadísticos por estado
    domain_stats: dict                  # {estado: {domain_id: {métrica: valor}}}
    timestamp: str
    segmentation_container: SegmentationContainer | None
    segmentation_method: SegmentationMethod | None
```

Se construye en `MainWindow.onSegmentationAccepted` y se pasa a `ResultsPanel`. Al cargar una sesión guardada, se reconstruye desde disco.

---

### Persistencia — `persistence/session_io.py`

Una sesión se guarda como un archivo `.session`, que es un ZIP con dos entradas:
- `session.json` — datos serializables (image_name, timestamp, parameters, domain_stats...).
- `arrays.npz` — arrays numpy (original_image, corrected_image, ising_result, labeled_images por estado).

Las claves enteras de los diccionarios se convierten a string antes de serializar (`keysToStr`) y se restauran al cargar (`strKeysToInt`), ya que JSON no soporta claves no-string.

**Funciones disponibles:**

| Función | Descripción |
|---|---|
| `saveSession(session, path)` | Guarda la sesión completa en `.session` |
| `loadSession(path)` | Carga y reconstruye una `Session` desde disco |
| `exportCorrectedImage(session, path)` | Escribe la imagen corregida como `.tif` con OpenCV |
| `exportDataExcel(session, path, min_area)` | Exporta métricas a `.xlsx`, una hoja por estado, filtrando por área |

---

## 9. Widgets especiales

### `DomainImageViewer` — `interface/visual_elements/domainImageViewer.py`

`QLabel` extendido que añade interactividad por clic sobre dominios. Mantiene internamente la imagen RGB coloreada y la imagen etiquetada (`labeled_image`) como arrays separados.

**Comportamiento:**
- Al hacer clic, convierte las coordenadas del widget a coordenadas de imagen (corrigiendo el offset y el factor de escala del escalado `KeepAspectRatio`).
- Consulta el `labeled_image` en esa posición para obtener el `domain_id`.
- Oscurece todos los dominios excepto el seleccionado (multiplicación por 0.25 sobre los píxeles no seleccionados).
- Emite `domain_clicked(domain_id: int, metrics: dict)`.

**API:**
```python
viewer = DomainImageViewer()
viewer.domain_clicked.connect(mi_slot)
viewer.setData(rgb_image, labeled_image, domain_stats)
```

---

## 10. Inputs y outputs

### Inputs

| Tipo | Formato | Dónde se usa |
|---|---|---|
| Imagen científica | `.tif`, `.tiff` (escala de grises, cualquier profundidad de bit) | `LoadPanel` → `Corrector` |
| Sesión guardada | `.session` (ZIP con JSON + NPZ) | `LoadPanel` → `loadSession` |

### Outputs

| Tipo | Formato | Cómo generarlo |
|---|---|---|
| Sesión completa | `.session` | Botón "Save" en `ResultsPanel` |
| Imagen corregida | `.tif` | Botón "Export Corrected Image" en `ResultsPanel` |
| Métricas de dominios | `.xlsx` (una hoja por estado) | Botón "Export Metrics" en `ResultsPanel` |

---

## English

## Table of Contents

1. [What is it and what does it aim for?](#1-what-is-it-and-what-does-it-aim-for)
2. [Architecture and data flow](#2-architecture-and-data-flow)
3. [Libraries and their purpose](#3-libraries-and-their-purpose)
4. [Directory structure](#4-directory-structure)
5. [How to add new functionality](#5-how-to-add-new-functionality)
6. [Current services](#6-current-services)
7. [Current panels](#7-current-panels)
8. [Data containers and persistence](#8-data-containers-and-persistence)
9. [Special widgets](#9-special-widgets)
10. [Inputs and outputs](#10-inputs-and-outputs)

---

## 1. What is it and what does it aim for?

Domain Analysis Tool is a desktop application for the quantitative analysis of scientific microscopy images, focused on the study of magnetic domains in XMCD images. The goal is to provide a tool that segments an image into intensity states, individually identifies and labels each domain within each state, and computes geometric metrics per domain.

**Design principles:**

- **Modularity** — each responsibility is isolated in its own module. Processing services do not know about the interface; panels do not know about the algorithms.
- **Extensibility** — adding a new segmentation method, a new metric, or a new panel should not require touching existing code beyond the integration point.
- **Layer separation** — processing, interface, persistence and statistics are independent layers that communicate through well-defined data containers (`Session`, `SegmentationContainer`).

---

## 2. Architecture and data flow

```
.tif image
    │
    ▼
Corrector                  ← processing/corrector.py
    │  (corrected_image: np.ndarray)
    ▼
Segmentation
    ├── Ising (ICM)         ← processing/isingMethodService.py
    └── GraphCutsService    ← processing/graphCutsService.py
    │  (SegmentationContainer)
    ▼
DomainService              ← processing/domainService.py
    │  (labeled_images, colored_images, domain_data)
    ▼
computeDomainStats         ← stats/domainStats.py
    │  (domain_stats: {state: {domain_id: {metric: value}}})
    ▼
Session                    ← core/session.py
    │  (container holding the full analysis)
    ▼
ResultsPanel               ← interface/panels/resultsPanel.py
    ├── DomainExplorerWindow
    └── Export (.session / .tif / .xlsx)
```

`Session` is the central object. It is built in `MainWindow` once segmentation is complete and passed between panels. Everything a panel needs is inside `Session`; panels do not call processing services directly (except `ResultsPanel`, which launches `DomainsWorker` in the background if the session was loaded from disk without pre-computed `domain_data`).

---

## 3. Libraries and their purpose

| Library | Use in the project |
|---|---|
| **PySide6** | GUI framework. All panels, windows and widgets inherit from Qt classes. |
| **numpy** | Image array operations (2D pixel matrices, masks, RGB images). Foundation of all processing. |
| **opencv-python** | Reading `.tif` images, normalization, circle detection (Hough), Gaussian blur for illumination correction. |
| **scikit-image** | `regionprops` for computing geometric metrics per domain; `label` for labeling connected regions. |
| **scikit-learn** | KMeans for Ising model initialization; GaussianMixture (GMM) for the data term in Graph Cuts. |
| **scipy** | `gaussian_kde` for the density curve in segmentation histograms. |
| **matplotlib** | Rendering histograms embedded in panels (via `FigureCanvasQTAgg`). |
| **openpyxl** | Writing `.xlsx` files for metrics export. |
| **seaborn** | Auxiliary statistical visualizations (used in exploration, not in the production interface). |
| **tifffile** | Reading TIFF images with greater fidelity than OpenCV for scientific multi-channel or high bit-depth formats. |
| **gco-wrapper** | Efficient Graph Cuts implementation (alpha-expansion) for graph-based segmentation. |

---

## 4. Directory structure

```
New_Tool/
│
├── main.py                        # Entry point
│
├── core/                          # Central data structures
│   ├── session.py                 # Session dataclass
│   ├── segmentationContainer.py   # SegmentationContainer dataclass + SegmentationMethod enum
│   └── pipeline.py                # PipelineDictator: headless orchestrator of the full flow
│
├── processing/                    # Processing services (no UI dependencies)
│   ├── corrector.py               # Illumination correction, stretching, brightness/contrast
│   ├── isingMethodService.py      # ICM segmentation (Ising model)
│   ├── graphCutsService.py        # Graph Cuts segmentation
│   └── domainService.py           # Domain labeling and per-domain data extraction
│
├── stats/                         # Metric computation
│   └── domainStats.py             # computeDomainStats: area, perimeter, roughness
│
├── persistence/                   # File reading and writing
│   └── session_io.py              # saveSession, loadSession, exportCorrectedImage, exportDataExcel
│
├── interface/                     # GUI layer
│   ├── styles.py                  # Color palette and global stylesheet
│   ├── mainWindow.py              # QMainWindow: panel orchestrator
│   │
│   ├── panels/                    # One file per screen/panel
│   │   ├── startPanel.py
│   │   ├── loadPanel.py
│   │   ├── bcPanel.py
│   │   ├── icmPanel.py
│   │   ├── graphCutsPanel.py
│   │   ├── resultsPanel.py
│   │   ├── domainExplorerWindow.py
│   │   └── domainComparisonPanel.py
│   │
│   └── visual_elements/           # Reusable widgets with their own logic
│       └── domainImageViewer.py
│
└── Elements/                      # External resources (fiji paths, temporary images)
```

**Separation logic:**
- `processing/` never imports from `interface/`. Services receive and return pure data (numpy arrays, dataclasses).
- `interface/` never implements processing logic. Panels call services and display results.
- `core/` defines structures shared across layers.
- `stats/` is independent: it receives labeled arrays and returns metric dictionaries.
- `persistence/` only reads and writes. It does not transform data.

---

## 5. How to add new functionality

### General service structure

A service in `processing/` is a class that receives data in the constructor or a `run()` method and exposes results via `get_data()` or `getContainer()`.

```python
# processing/myService.py
import numpy as np
from core.segmentationContainer import SegmentationContainer

class MyService:
    def __init__(self, param1: float, param2: int):
        self.param1 = param1
        self.param2 = param2

    def run(self, image: np.ndarray) -> None:
        # processing
        self.result = ...

    def getSegmentationContainer(self) -> SegmentationContainer:
        return SegmentationContainer(
            original_image=...,
            mask=...,
            final_image=self.result,
            num_states=...,
            parameters=...,
            initial_labels=...,
            method=SegmentationMethod.MY_METHOD,
        )
```

If the method is a new segmentation type, add its entry to the `SegmentationMethod` enum in `core/segmentationContainer.py`:

```python
class SegmentationMethod(Enum):
    ICM = "ICM"
    GRAPH_CUTS = "GraphCuts"
    MY_METHOD = "MyMethod"   # new
```

### General panel structure

A panel is a `QWidget` that:
- Emits signals to communicate with `MainWindow` (never calls other panels directly).
- Contains a `QThread` worker if it launches heavy processing in the background.
- Separates UI construction (`buildUi`) from data logic (`loadData`, `onWorkerFinished`).

```python
# interface/panels/myPanel.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal

class MyPanel(QWidget):
    result_accepted = Signal(object)   # emits result to MainWindow
    cancelled = Signal()
    home = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.buildUi()

    def buildUi(self):
        layout = QVBoxLayout(self)
        self.run_but = QPushButton("Run")
        self.run_but.clicked.connect(self.onRunClicked)
        layout.addWidget(self.run_but)

    def loadData(self, data):
        # receives data from MainWindow and prepares the panel
        self.data = data

    def onRunClicked(self):
        # launches processing and emits signal with result
        result = ...
        self.result_accepted.emit(result)
```

If processing is expensive, use a worker:

```python
from PySide6.QtCore import QThread, Signal

class MyWorker(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        try:
            result = MyService().run(self.data)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

### Integrating the new panel in MainWindow

`MainWindow` uses a `QStackedWidget` to manage screens. To add a new panel:

```python
# interface/mainWindow.py

# 1. Import
from interface.panels.myPanel import MyPanel

# 2. In buildUi(), instantiate and connect signals
self.my_panel = MyPanel()
self.my_panel.result_accepted.connect(self.onMyResult)
self.my_panel.cancelled.connect(self.goToStart)
self.my_panel.home.connect(self.goToStart)
self.stack.addWidget(self.my_panel)

# 3. Navigation methods
def goToMyPanel(self):
    self.my_panel.loadData(self.required_data)
    self.stack.setCurrentWidget(self.my_panel)

def onMyResult(self, result):
    # store result in self.session and navigate to the next step
    self.session.my_data = result
    self.goToResults()
```

---

## 6. Current services

### `Corrector` — `processing/corrector.py`

Image correction in four chained steps:

1. **`create_mask`** — detects the circular region of interest via `HoughCircles`. If no circle is detected, the mask covers the whole image.
2. **`correct_illumination`** — corrects illumination gradients by subtracting a background estimated with Gaussian blur.
3. **`linear_stretch`** — stretches the histogram to [0, 255] between `v_low` and `v_high`.
4. **`adjust_brightness_contrast`** — applies a brightness offset and a contrast factor.

The main method is `apply_correction`, which chains all steps. Individual steps are public and can be called separately.

---

### `Ising` — `processing/isingMethodService.py`

Bayesian segmentation using the Ising model with ICM (Iterated Conditional Modes) optimization:

1. **Initialization** with KMeans (`n_clusters = num_states`) over the pixels inside the mask.
2. **ICM** — each iteration minimizes per-pixel energy: `E = statistical_energy + neighborhood_energy`. Statistical energy measures the distance of each pixel to the mean of each state (Gaussian distribution); neighborhood energy encourages adjacent pixels to share the same state (controlled by `beta`).
3. **Convergence** — the algorithm stops when no pixel changes state or `max_iterations` is reached.

Exposes `getSegmentationContainer()` to build the standard container.

---

### `GraphCutsService` — `processing/graphCutsService.py`

Graph Cuts segmentation with alpha-expansion:

1. **GMM** — models the intensity distribution of each state with a mixture of Gaussians (`GaussianMixture`, `number_gaussians_per_state` components).
2. **Data term** — negative log-probability of each pixel under each state according to the GMM.
3. **Smoothness term (N-link)** — penalizes adjacent pixels with different states, weighted by `lambda` and the intensity difference (controlled by `sigma`).
4. **Alpha-expansion** — global energy minimization via `gco-wrapper`.

Exposes `getSegmentationContainer()` with the same contract as `Ising`.

---

### `DomainService` — `processing/domainService.py`

Starting from a `SegmentationContainer`, extracts individual domains:

1. **`extract_state_images`** — generates a binary image per state (1 inside the state and inside the mask, 0 outside).
2. **`label_domains`** — applies `skimage.measure.label` with 8-connectivity to label connected regions. Each domain receives a unique id per state.
3. **`color_domains`** — assigns a random color to each domain for visualization.
4. **`extract_domain_data`** — extracts coordinates and intensity values per domain.

The `get_data()` method returns a dictionary with all these structures.

---

### `computeDomainStats` — `stats/domainStats.py`

Computes geometric metrics for each domain using `skimage.measure.regionprops`. Iterates over labeled images and builds the structure:

```
{ state: { domain_id: { "area": float, "perimeter": float, "roughness": float } } }
```

Each metric function is independent and receives the `regionprops` `domain` object. See the "How to add a new metric" section in the user README.

---

## 7. Current panels

### `StartPanel` — `interface/panels/startPanel.py`

Start screen. Emits `analyse_requested` to go to the analysis flow or `compare_requested` to go to session comparison.

---

### `LoadPanel` — `interface/panels/loadPanel.py`

Loads a `.tif` image with OpenCV or a `.session` file via `loadSession`. Allows selecting the segmentation method (ICM / Graph Cuts) before continuing. Emits `image_loaded(np.ndarray, str)` or `session_loaded(Session)`.

---

### `BcPanel` — `interface/panels/bcPanel.py`

Brightness and contrast correction with real-time preview. Calls `Corrector` directly without a worker (correction is fast). Emits `correction_accepted(np.ndarray, int, int)` with the corrected image and v_low / v_high values.

---

### `IsingPanel` — `interface/panels/icmPanel.py`

ICM segmentation. Launches `IsingWorker` (QThread) with the configured parameters. Displays the original image, the view for the selected state, the intensity histogram, and the state distribution legend. Emits `segmentation_accepted(SegmentationContainer)`.

---

### `GraphCutsPanel` — `interface/panels/graphCutsPanel.py`

Graph Cuts segmentation. Same visual structure as `IsingPanel` with its own parameters (Lambda, Sigma, Gaussians, Iterations). Launches `GraphCutsWorker`. Emits `segmentation_accepted(SegmentationContainer)`.

---

### `ResultsPanel` — `interface/panels/resultsPanel.py`

Main results panel. Receives a `Session` and displays:
- Original image and colored domain view.
- Histogram of the active metric per state.
- State tabs and metric buttons.
- Area filter and histogram range controls.

If the session has a `segmentation_container` but no `domain_stats`, it launches `DomainsWorker` in the background to compute them. From here, `DomainExplorerWindow` is opened and exports are accessed.

---

### `DomainExplorerWindow` — `interface/panels/domainExplorerWindow.py`

Auxiliary window (`QDialog`) for inspecting domains individually. Uses `DomainImageViewer` for the interactive image and a `QTableWidget` to display the selected domain's metrics alongside the state mean and global mean, with the deviation percentage.

---

### `DomainComparisonPanel` — `interface/panels/domainComparisonPanel.py`

Allows loading two `.session` files and comparing their metrics side by side.

---

## 8. Data containers and persistence

### `SegmentationContainer` — `core/segmentationContainer.py`

Dataclass that encapsulates the result of any segmentation method:

```python
@dataclass
class SegmentationContainer:
    original_image: np.ndarray      # normalized input image
    mask: np.ndarray                # boolean mask of the region of interest
    final_image: np.ndarray         # segmented image (values 0..num_states-1)
    num_states: int                 # number of states
    parameters: dict                # {state: {"mean": float, "std": float}}
    initial_labels: np.ndarray      # initial labels (before optimization)
    method: SegmentationMethod      # ICM or GRAPH_CUTS
    method_configuration: dict      # parameters used (beta, lambda, etc.)
```

It is the contract between segmentation services and the rest of the system. `DomainService` consumes it directly.

---

### `Session` — `core/session.py`

Central dataclass that accumulates the full analysis state:

```python
@dataclass
class Session:
    image_name: str
    original_image: np.ndarray | None
    corrected_image: np.ndarray | None
    ising_result: np.ndarray | None
    domain_data: dict                   # output of DomainService.get_data()
    parameters: dict                    # configuration used
    ising_stats: dict                   # statistical parameters per state
    domain_stats: dict                  # {state: {domain_id: {metric: value}}}
    timestamp: str
    segmentation_container: SegmentationContainer | None
    segmentation_method: SegmentationMethod | None
```

Built in `MainWindow.onSegmentationAccepted` and passed to `ResultsPanel`. When loading a saved session, it is reconstructed from disk.

---

### Persistence — `persistence/session_io.py`

A session is saved as a `.session` file, which is a ZIP with two entries:
- `session.json` — serializable data (image_name, timestamp, parameters, domain_stats...).
- `arrays.npz` — numpy arrays (original_image, corrected_image, ising_result, labeled_images per state).

Integer dictionary keys are converted to strings before serialization (`keysToStr`) and restored on load (`strKeysToInt`), since JSON does not support non-string keys.

**Available functions:**

| Function | Description |
|---|---|
| `saveSession(session, path)` | Saves the full session to `.session` |
| `loadSession(path)` | Loads and reconstructs a `Session` from disk |
| `exportCorrectedImage(session, path)` | Writes the corrected image as `.tif` using OpenCV |
| `exportDataExcel(session, path, min_area)` | Exports metrics to `.xlsx`, one sheet per state, filtered by area |

---

## 9. Special widgets

### `DomainImageViewer` — `interface/visual_elements/domainImageViewer.py`

Extended `QLabel` that adds click interactivity over domains. Internally holds the colored RGB image and the labeled image (`labeled_image`) as separate arrays.

**Behavior:**
- On click, converts widget coordinates to image coordinates (correcting for offset and the `KeepAspectRatio` scale factor).
- Looks up `labeled_image` at that position to get the `domain_id`.
- Dims all domains except the selected one (pixel multiplication by 0.25 on non-selected pixels).
- Emits `domain_clicked(domain_id: int, metrics: dict)`.

**API:**
```python
viewer = DomainImageViewer()
viewer.domain_clicked.connect(my_slot)
viewer.setData(rgb_image, labeled_image, domain_stats)
```

---

## 10. Inputs and outputs

### Inputs

| Type | Format | Where it is used |
|---|---|---|
| Scientific image | `.tif`, `.tiff` (grayscale, any bit depth) | `LoadPanel` → `Corrector` |
| Saved session | `.session` (ZIP with JSON + NPZ) | `LoadPanel` → `loadSession` |

### Outputs

| Type | Format | How to generate it |
|---|---|---|
| Full session | `.session` | "Save" button in `ResultsPanel` |
| Corrected image | `.tif` | "Export Corrected Image" button in `ResultsPanel` |
| Domain metrics | `.xlsx` (one sheet per state) | "Export Metrics" button in `ResultsPanel` |

## 11. A a A
