import numpy as np
import sklearn.cluster as cluster
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns

def upload_image():
    path = r"C:\Users\user\Desktop\TFG-Teleco\New_Tool\Images ALBA - Sample PyHM004\Low temperature\DICHROISM_fiji_ready.tif"
    try:
        # cv.IMREAD_GRAYSCALE fuerza a OpenCV a leer el PNG directamente en 2D (escala de grises)
        y = cv.imread(path, cv.IMREAD_GRAYSCALE)
        
        # Control de seguridad por si la ruta está mal o el archivo está corrupto
        if y is None:
            print(f"Error: No se pudo encontrar o cargar la imagen en la ruta: {path}")
            return None
            
        # Ya sabemos 100% que es 2D y float32
        y = y.astype(np.float32)
        
        # Normalización
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 255.0
        
        return y
        
    except Exception as e:
        print(f"Error al leer el archivo de imagen: {e}")
        return None

def upload_image2():
    ruta = r"C:\Users\user\Desktop\TFG-Teleco\New_Tool\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
    try:
        y = tiff.imread(ruta)
        if len(y.shape) == 3:
            y = np.mean(y, axis=2)
        y = y.astype(np.float32)
        
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 255.0
        return y
        
    except Exception as e:
        print(f"Error al leer el archivo TIFF: {e}")
        return None
    
def get_circular_mask(y):
    y_8bit = cv.normalize(y, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    blurred = cv.medianBlur(y_8bit, 5)
    
    circles = cv.HoughCircles(
        blurred, cv.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=int(y.shape[0]*0.3), maxRadius=int(y.shape[0]*0.6)
    )
    
    mask = np.zeros(y.shape, dtype=bool)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x_c, y_c, r = circles[0, 0]
        
        mask_img = np.zeros(y.shape, dtype=np.uint8)
        cv.circle(mask_img, (x_c, y_c), r, 255, -1) # -1 para rellenar
        
        mask = mask_img == 255
        print(f"Máscara circular creada: Centro({x_c}, {y_c}), Radio({r})")
    else:
        print("No se detectó el círculo, se usará toda la imagen.")
        mask = np.ones(y.shape, dtype=bool)
        
    return mask

def calculate_statistical_variables(y, x, num_states, mask=None):
    parameters = {}
    for state in range(num_states):
        if mask is not None: 
            pixel_state_values = y[(x==state) & mask]
        else: 
            pixel_state_values = y[x==state]
        if len(pixel_state_values) > 0:
            parameters[state] = {
                'mean': np.mean(pixel_state_values),
                'std': np.std(pixel_state_values)+1e-6
            }
        else:
            print(f"Advertencia: No se encontraron píxeles para el estado {state}. Se asignarán valores por defecto.")
            parameters[state] = {
                'mean': 0,
                'std': 1e-6
            }
    return parameters

def initialize_ising_model(y, num_states, mask=None):
    if mask is not False and mask is None:
        mask = get_circular_mask(y)
    y_kmeans = y[mask].reshape(-1, 1)
    kmeans = cluster.KMeans(n_clusters=num_states, n_init=10, random_state=64)
    x_labels = kmeans.fit_predict(y_kmeans)
    x_matrix = np.full(y.shape, -1, dtype=np.int32)
    x_matrix[mask] = x_labels
    parameters = calculate_statistical_variables(y, x_matrix, num_states, mask)
    return x_matrix, parameters

def calculate_neighbour_sum_vectorized(x, state):
    match = (x == state).astype(np.float32)
    neighbour_sum = (
        np.roll(match, 1, axis=0) +   # arriba
        np.roll(match, -1, axis=0) +  # abajo
        np.roll(match, 1, axis=1) +   # izquierda
        np.roll(match, -1, axis=1)    # derecha
    )
    return neighbour_sum

def apply_ising_model_icm_fast(y, x, parameters, num_states, beta, max_iterations, mask):
    x_iter = x.copy()
    
    for iteration in range(max_iterations):
        x_old = x_iter.copy()
        energy_maps = np.full((num_states, *y.shape), np.inf)
        
        for state in range(num_states):
            mu = parameters[state]['mean']
            sigma = parameters[state]['std']
            
            stat_energy = ((y - mu) / sigma) ** 2
            neigh_energy = -beta * calculate_neighbour_sum_vectorized(x_iter, state)
            energy_maps[state] = stat_energy + neigh_energy
        
        x_new = np.argmin(energy_maps, axis=0).astype(np.int32)
        x_iter = np.where(mask, x_new, x_iter)
        
        if np.array_equal(x_iter, x_old) and iteration > 0:
            print(f"Converged at iteration {iteration}")
            break
        
        parameters = calculate_statistical_variables(y, x_iter, num_states, mask)
    
    return x_iter

def plot_analysis(y, x_final, mask):
    """Genera las gráficas de porcentaje de ocupación e histogramas de distribución de intensidad"""
    # Filtrar únicamente las zonas activas por la máscara
    y_valid = y[mask]
    x_valid = x_final[mask]
    total_pixels = len(x_valid)
    
    states = [0, 1, 2]
    # Códigos Hexadecimales para Matplotlib (Negro-Azul, Gris-Verde, Blanco-Rojo)
    colors_hex = ['#0A192F', '#7A8B7B', '#E65C5C'] 
    labels = ['Estado 0 (Negro-Azul)', 'Estado 1 (Gris-Verde)', 'Estado 2 (Blanco-Rojo)']
    
    # 1. Calcular porcentajes
    percentages = []
    for state in states:
        count = np.sum(x_valid == state)
        percentages.append((count / total_pixels) * 100)
        
    # Crear ventana de gráficos
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.canvas.manager.set_window_title('Análisis Estadístico del Modelo de Ising')
    
    # GRÁFICA 1: Porcentajes de cada estado
    bars = axes[0].bar(labels, percentages, color=colors_hex, edgecolor='black', linewidth=1.2)
    axes[0].set_title('Porcentaje de Ocupación por Estado', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Porcentaje (%)')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir texto con el porcentaje exacto encima de cada barra
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height:.2f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  
                         textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')
                         
    # GRÁFICA 2: Histograma e Intensidad de Distribución (Seaborn)
    for state, color, label in zip(states, colors_hex, labels):
        state_intensities = y_valid[x_valid == state]
        if len(state_intensities) > 0:
            sns.histplot(state_intensities, bins=50, kde=True, ax=axes[1], 
                         color=color, label=label, alpha=0.6, element="step", stat="density")
                         
    axes[1].set_title('Distribución de Intensidad de Píxeles', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Valor de gris original (0 - 255)')
    axes[1].set_ylabel('Densidad de frecuencia')
    axes[1].grid(linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
y = upload_image()

# Control de interrupción: Si 'y' no se leyó, detenemos el flujo para evitar caídas
if y is None:
    print("No se puede continuar el proceso porque la imagen no fue cargada correctamente.")
else:
    mask = np.ones_like(y, dtype=bool) # Máscara completa (Rectangular)
    
    x_init, params = initialize_ising_model(y, num_states=3, mask=mask)
    x_final = apply_ising_model_icm_fast(y, x_init, params, num_states=3, beta=3, max_iterations=30, mask=mask)
    
    with open("script_resut.pkl", "wb") as f:
        import pickle 
        pickle.dump(x_final, f)

    # --- Construcción de Imagen de Estados Personalizada a Color para OpenCV ---
    h, w = x_final.shape
    color_output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Configuración de colores en formato BGR (OpenCV funciona al revés: Blue, Green, Red)
    # Estado 0: Negro Azul -> Azul muy oscuro con toque negro
    color_output[x_final == 0] = [245, 215, 190]    
    # Estado 1: Gris Verde -> Tono grisáceo oliva
    color_output[x_final == 1] = [0, 0, 0]
    # Estado 2: Blanco Rojo -> Rojo claro o brillante satinado
    color_output[x_final == 2] = [50, 255, 205]
    
    # Aplicar máscara (por si en el futuro cambias a circular)
    color_output[~mask] = [0, 0, 0]

    # Visualización mediante OpenCV
    cv.namedWindow('Original (Escala de Grises)', cv.WINDOW_NORMAL)
    cv.imshow('Original (Escala de Grises)', y.astype(np.uint8))
    cv.resizeWindow('Original (Escala de Grises)', 400, 400)
    
    cv.namedWindow('K-means Inicial', cv.WINDOW_NORMAL)
    cv.imshow('K-means Inicial', (x_init * 120).astype(np.uint8))
    cv.resizeWindow('K-means Inicial', 400, 400)
    
    cv.namedWindow('Resultado ICM Grises', cv.WINDOW_NORMAL)
    cv.imshow('Resultado ICM Grises', (x_final * (255 // (3 - 1))).astype(np.uint8))
    cv.resizeWindow('Resultado ICM Grises', 400, 400)
    
    # NUEVA VENTANA: Muestra el resultado final usando tus especificaciones de color
    cv.namedWindow('Resultado ICM Personalizado Color', cv.WINDOW_NORMAL)
    cv.imshow('Resultado ICM Personalizado Color', color_output)
    cv.resizeWindow('Resultado ICM Personalizado Color', 400, 400)
    
    # Lanzar la ventana de gráficos estadísticos (Se ejecuta en paralelo al waitKey)
    plot_analysis(y, x_final, mask)
    
    cv.waitKey(0)
    cv.destroyAllWindows()