# Domain Analysis Tool

**[Español](#español) | [English](#english)**

---

## Español

### ¿Qué es?

Domain Analysis Tool es una aplicación de escritorio para el análisis de imágenes científicas de microscopía. Permite segmentar dominios en una imagen, calcular métricas por dominio y estado, y exportar los resultados.

### Requisitos

- Python 3.12 (recomendado; también funciona con 3.11)
- Las librerías listadas en `requirements.txt`

### Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/Alaldb/TFG-Telecomunications-Sincotron-Images-Processing.git
cd TFG-Telecomunications-Sincotron-Images-Processing
```

2. Crea y activa un entorno aislado:

**Opción A — venv (Python estándar)**
```bash
python3.12 -m venv dAT
source dAT/bin/activate        # Linux / macOS
dAT\Scripts\activate           # Windows
```

**Opción B — Anaconda / Miniconda**
```bash
conda create -n domain-analysis python=3.12
conda activate domain-analysis
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecuta la aplicación:
```bash
python main.py
```

### Flujo de uso

**1. Pantalla de inicio**
Al abrir la aplicación aparecen dos opciones: iniciar un nuevo análisis o comparar sesiones existentes.

**2. Cargar imagen o sesión**
- *Load image* — carga una imagen en formato `.tif` / `.tiff` para iniciar un nuevo análisis.
- *Load session* — carga un archivo `.session` guardado previamente para retomar o revisar un análisis.
- Selecciona el método de segmentación: **ICM** o **Graph Cuts**.

**3. Corrección de brillo y contraste**
Ajusta el rango de intensidad de la imagen antes de segmentar. Puedes modificar brillo, contraste y recorte de bordes. La corrección se previsualiza en tiempo real.

**4. Segmentación**
Configura los parámetros del método elegido y lanza la segmentación. La imagen se divide en estados (niveles de intensidad) que representan los distintos dominios.

**5. Resultados**
Una vez completada la segmentación se muestra el panel de resultados con:
- Vista de la imagen original y de los dominios segmentados con colores.
- Histograma de distribución de la métrica activa.
- Pestañas para navegar entre estados o ver todos a la vez.
- Botones para cambiar la métrica visualizada en el histograma.
- **Filtro de área mínima** — excluye del análisis los dominios más pequeños que el valor indicado.
- **Rango del histograma** — ajusta el eje X del histograma manualmente.

**6. Domain Explorer**
Abre una ventana auxiliar para explorar cada dominio individualmente con sus métricas detalladas.

**7. Comparación de sesiones**
Desde la pantalla de inicio puedes cargar dos sesiones y comparar sus métricas lado a lado.

### Exportar resultados

Desde el panel de resultados dispones de tres opciones de exportación:

| Botón | Resultado | Formato |
|---|---|---|
| Save | Guarda la sesión completa para retomarla más adelante | `.session` |
| Export Corrected Image | Exporta la imagen con la corrección aplicada | `.tif` |
| Export Metrics | Exporta las métricas de todos los dominios | `.xlsx` |

> Si hay un filtro de área activo, el Excel solo incluirá los dominios que superen dicho filtro.

### Formato del Excel exportado

El archivo `.xlsx` contiene una hoja por cada estado. Cada hoja tiene una fila por dominio con su ID y todas las métricas disponibles como columnas.

Ejemplo:

| DOMAIN ID | AREA | PERIMETER | ROUGHNESS |
|---|---|---|---|
| 1 | 124.0 | 52.34 | 1.16 |
| 2 | 87.0 | 38.97 | 1.38 |

### Métricas calculadas

Las métricas se calculan automáticamente para cada dominio usando `skimage.measure.regionprops`:

- **Area** — número de píxeles del dominio.
- **Perimeter** — longitud del contorno en píxeles.
- **Roughness** — irregularidad del contorno, calculada como P² / (4π·A). Un valor de 1.0 corresponde a un círculo perfecto; valores mayores indican contornos más irregulares.

> Los dominios de 1 píxel pueden tener perímetro 0. Se recomienda usar el filtro de área mínima para excluirlos.

### Referencia de parámetros

#### Corrección de brillo y contraste

| Parámetro | Rango | Descripción |
|---|---|---|
| Low intensity end | 0 – 255 | Valor mínimo del rango de intensidad. Los píxeles por debajo de este valor se recortan a negro. |
| High intensity end | 0 – 255 | Valor máximo del rango de intensidad. Los píxeles por encima se recortan a blanco. |
| Brightness | -127 – 127 | Desplazamiento aditivo sobre todos los píxeles. 0 no aplica cambio. |
| Contrast | ≥ 0.1 | Factor multiplicativo sobre la imagen. 1.0 no aplica cambio. Usar `.` como separador decimal. |
| Crop borders | 0 – 100 px | Recorta N píxeles de cada borde antes del procesado. Útil si los bordes introducen artefactos en la segmentación. |

#### Segmentación — ICM

| Parámetro | Rango | Descripción |
|---|---|---|
| Beta | ≥ 0.0 | Regularización espacial. Valores más altos producen bordes más suaves entre estados. Rango típico: 0.5 – 5. |
| Max Iterations | 1 – 200 | Número máximo de iteraciones. El algoritmo para antes si converge. |
| Num States | 2 – 8 | Número de estados de intensidad a segmentar. 3 es el valor habitual para imágenes de dominios magnéticos. |

#### Segmentación — Graph Cuts

| Parámetro | Rango | Descripción |
|---|---|---|
| Num States | 2 – 8 | Número de estados de intensidad a segmentar. |
| Lambda | ≥ 0.0 | Peso de suavizado espacial. Equivalente al Beta del ICM. Rango típico: 0.01 – 5. |
| Sigma | ≥ 0.0 o vacío | Sensibilidad del N-link a diferencias de intensidad. Vacío = estimación automática. Valores menores detectan bordes más finos. |
| Iterations | -1 – 200 | Número máximo de rondas de alpha-expansion. -1 ejecuta hasta convergencia (recomendado). |
| Gaussians | 1 – 10 | Componentes Gaussianas por estado en el modelo GMM. Valores más altos modelan distribuciones complejas a costa de velocidad. |

#### Panel de resultados

| Control | Descripción |
|---|---|
| Pestañas de estado | Filtra la vista y el histograma al estado seleccionado. "All States" muestra todos a la vez. |
| Botones de métrica | Selecciona qué métrica se representa en el histograma. |
| Minimum area | Excluye del análisis los dominios con área inferior al valor indicado (px). Afecta también al Excel exportado. |
| Histogram range (Min / Max) | Fija el rango del eje X del histograma. 0 en ambos = escala automática. |

---

### Cómo añadir una nueva métrica

Las métricas se definen en `stats/domainStats.py`. Para añadir una nueva:

1. Define una función que reciba un objeto `domain` de `regionprops` y devuelva un `float`:
```python
def computeCircularity(domain) -> float:
    area = computeArea(domain)
    perimeter = computePerimeter(domain)
    if perimeter > 0:
        return (4 * math.pi * area) / (perimeter ** 2)
    return 0.0
```

2. Añádela al diccionario `metrics` dentro de `computeDomainStats`:
```python
metrics: dict[str, float] = {
    "area":        computeArea(domain),
    "perimeter":   computePerimeter(domain),
    "roughness":   computeRoughness(domain),
    "circularity": computeCircularity(domain),  # nueva métrica
}
```

La nueva métrica aparecerá automáticamente en el histograma, en el Domain Explorer y en el Excel exportado sin ningún cambio adicional.

---

## English

### What is it?

Domain Analysis Tool is a desktop application for the analysis of scientific microscopy images. It allows you to segment domains in an image, compute per-domain and per-state metrics, and export the results.

### Requirements

- Python 3.12 (recommended; 3.11 also works)
- Libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Alaldb/TFG-Telecomunications-Sincotron-Images-Processing.git
cd TFG-Telecomunications-Sincotron-Images-Processing
```

2. Create and activate an isolated environment:

**Option A — venv (standard Python)**
```bash
python3.12 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**Option B — Anaconda / Miniconda**
```bash
conda create -n domain-analysis python=3.12
conda activate domain-analysis
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

### Usage

**1. Start screen**
When the application opens, two options are available: start a new analysis or compare existing sessions.

**2. Load image or session**
- *Load image* — loads a `.tif` / `.tiff` image to start a new analysis.
- *Load session* — loads a previously saved `.session` file to resume or review an analysis.
- Select the segmentation method: **ICM** or **Graph Cuts**.

**3. Brightness and contrast correction**
Adjust the intensity range of the image before segmentation. You can modify brightness, contrast, and edge cropping. The correction is previewed in real time.

**4. Segmentation**
Configure the parameters of the chosen method and run the segmentation. The image is divided into states (intensity levels) representing the different domains.

**5. Results**
Once segmentation is complete, the results panel is shown with:
- View of the original image and the segmented domains in color.
- Distribution histogram of the active metric.
- Tabs to navigate between states or view all at once.
- Buttons to switch the metric displayed in the histogram.
- **Minimum area filter** — excludes domains smaller than the specified value from the analysis.
- **Histogram range** — manually adjusts the X axis of the histogram.

**6. Domain Explorer**
Opens an auxiliary window to inspect each domain individually with detailed metrics.

**7. Session comparison**
From the start screen you can load two sessions and compare their metrics side by side.

### Exporting results

From the results panel, three export options are available:

| Button | Output | Format |
|---|---|---|
| Save | Saves the full session to resume later | `.session` |
| Export Corrected Image | Exports the image with the applied correction | `.tif` |
| Export Metrics | Exports the metrics for all domains | `.xlsx` |

> If a minimum area filter is active, the Excel file will only include domains that exceed that filter.

### Excel export format

The `.xlsx` file contains one sheet per state. Each sheet has one row per domain with its ID and all available metrics as columns.

Example:

| DOMAIN ID | AREA | PERIMETER | ROUGHNESS |
|---|---|---|---|
| 1 | 124.0 | 52.34 | 1.16 |
| 2 | 87.0 | 38.97 | 1.38 |

### Computed metrics

Metrics are computed automatically for each domain using `skimage.measure.regionprops`:

- **Area** — number of pixels in the domain.
- **Perimeter** — contour length in pixels.
- **Roughness** — contour irregularity, computed as P² / (4π·A). A value of 1.0 corresponds to a perfect circle; higher values indicate more irregular contours.

> Domains of 1 pixel may have a perimeter of 0. It is recommended to use the minimum area filter to exclude them.

### Parameter reference

#### Brightness and contrast correction

| Parameter | Range | Description |
|---|---|---|
| Low intensity end | 0 – 255 | Minimum intensity value. Pixels below this are clipped to black. |
| High intensity end | 0 – 255 | Maximum intensity value. Pixels above this are clipped to white. |
| Brightness | -127 – 127 | Additive offset applied to all pixels. 0 means no change. |
| Contrast | ≥ 0.1 | Multiplicative factor applied to the image. 1.0 means no change. Use `.` as decimal separator. |
| Crop borders | 0 – 100 px | Removes N pixels from each border before processing. Useful when borders introduce segmentation artifacts. |

#### Segmentation — ICM

| Parameter | Range | Description |
|---|---|---|
| Beta | ≥ 0.0 | Spatial regularization. Higher values produce smoother boundaries between states. Typical range: 0.5 – 5. |
| Max Iterations | 1 – 200 | Maximum number of iterations. The algorithm stops earlier if it converges. |
| Num States | 2 – 8 | Number of intensity states to segment. 3 is typical for magnetic domain images. |

#### Segmentation — Graph Cuts

| Parameter | Range | Description |
|---|---|---|
| Num States | 2 – 8 | Number of intensity states to segment. |
| Lambda | ≥ 0.0 | Spatial smoothness weight. Equivalent to Beta in ICM. Typical range: 0.01 – 5. |
| Sigma | ≥ 0.0 or empty | Sensitivity of the N-link to intensity differences. Empty = automatic estimation. Lower values detect finer edges. |
| Iterations | -1 – 200 | Maximum number of alpha-expansion rounds. -1 runs until convergence (recommended). |
| Gaussians | 1 – 10 | Gaussian components per state in the GMM model. Higher values model complex distributions at the cost of speed. |

#### Results panel

| Control | Description |
|---|---|
| State tabs | Filters the view and histogram to the selected state. "All States" shows all at once. |
| Metric buttons | Selects which metric is displayed in the histogram. |
| Minimum area | Excludes domains with area below the specified value (px) from the analysis. Also affects the exported Excel. |
| Histogram range (Min / Max) | Sets the X axis range of the histogram. 0 in both fields = automatic scale. |

---

### How to add a new metric

Metrics are defined in `stats/domainStats.py`. To add a new one:

1. Define a function that receives a `regionprops` `domain` object and returns a `float`:
```python
def computeCircularity(domain) -> float:
    area = computeArea(domain)
    perimeter = computePerimeter(domain)
    if perimeter > 0:
        return (4 * math.pi * area) / (perimeter ** 2)
    return 0.0
```

2. Add it to the `metrics` dictionary inside `computeDomainStats`:
```python
metrics: dict[str, float] = {
    "area":        computeArea(domain),
    "perimeter":   computePerimeter(domain),
    "roughness":   computeRoughness(domain),
    "circularity": computeCircularity(domain),  # new metric
}
```

The new metric will automatically appear in the histogram, the Domain Explorer, and the exported Excel file with no further changes.
