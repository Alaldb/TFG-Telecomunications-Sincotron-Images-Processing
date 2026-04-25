import numpy as np
import sklearn.cluster as cluster
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt

def upload_image():
    ruta = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
    try:
        y = tiff.imread(ruta)
        if len(y.shape) == 3:
            y = np.mean(y, axis=2)
        
        y = y.astype(np.float32)
        # Normalización robusta para evitar que outliers rompan el histograma
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
        cv.circle(mask_img, (x_c, y_c), r, 255, -1)
        mask = mask_img == 255
        print(f"Máscara circular creada: Centro({x_c}, {y_c}), Radio({r})")
    else:
        print("No se detectó el círculo, se usará toda la imagen.")
        mask = np.ones(y.shape, dtype=bool)
    return mask

def calculate_statistical_variables(y, x, num_states, mask):
    parameters = {}
    for state in range(num_states):
        # CORRECCIÓN: Paréntesis obligatorios para que & funcione bien
        condicion = (x == state) & mask
        pixel_state_values = y[condicion]
        
        if len(pixel_state_values) > 100:
            parameters[state] = {
                'mean': np.mean(pixel_state_values),
                'std': np.std(pixel_state_values) + 2.0  # Suelo de ruido para evitar colapso
            }
        else:
            # Valores por defecto si una clase se queda vacía
            parameters[state] = {'mean': (state+1)*(255/(num_states+1)), 'std': 15.0}
    return parameters

def initialize_ising_model(y, num_states, mask):
    y_kmeans = y[mask].reshape(-1, 1)
    kmeans = cluster.KMeans(n_clusters=num_states, n_init=10, random_state=64)
    x_labels = kmeans.fit_predict(y_kmeans)
    
    # Fondo inicializado a -1 para que no cuente como estado 0
    x_matrix = np.full(y.shape, -1, dtype=np.int32)
    x_matrix[mask] = x_labels
    
    parameters = calculate_statistical_variables(y, x_matrix, num_states, mask)
    return x_matrix, parameters

def calculate_energy(y_value, x, mu, sigma, beta, state, row, col):
    # Energía Gaussiana completa (Log-likelihood negativa)
    # El término np.log(sigma) es clave para evitar que todos los píxeles vayan a una sola clase
    dist_energy = 0.5 * ((y_value - mu) / sigma)**2 + np.log(sigma)
    
    # Energía de vecindad (Ising)
    neighbour_sum = 0
    mov_tuples = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = x.shape
    for dx, dy in mov_tuples:
        nx, ny = row + dx, col + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            if x[nx, ny] == state:
                neighbour_sum += 1
    
    return dist_energy - (beta * neighbour_sum)

def apply_ising_model_icm(y, x, parameters, num_states, beta, max_iterations, mask):
    rows, cols = y.shape
    x_iteration = x.copy()
    
    for iteration in range(max_iterations):
        x_old = x_iteration.copy()
        
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]: continue
                
                best_energy = float('inf')
                best_state = x_iteration[r, c]
                
                for state in range(num_states):
                    energy = calculate_energy(y[r, c], x_iteration, 
                                              parameters[state]['mean'], 
                                              parameters[state]['std'], 
                                              beta, state, r, c)
                    if energy < best_energy:
                        best_energy = energy
                        best_state = state
                
                x_iteration[r, c] = best_state
        
        # Reporte de estado por iteración
        print(f"Iteración {iteration+1} finalizada.")
        for s in range(num_states):
            n_pix = np.sum((x_iteration == s) & mask)
            print(f"  Estado {s}: {n_pix} px | Media: {parameters[s]['mean']:.2f}")

        if np.array_equal(x_iteration, x_old):
            print("Convergencia alcanzada.")
            break
            
        parameters = calculate_statistical_variables(y, x_iteration, num_states, mask)
    
    return x_iteration

# --- EJECUCIÓN PRINCIPAL ---
y = upload_image()
if y is not None:
    mask = get_circular_mask(y)
    num_states = 3
    
    # Inicialización
    x_init, params = initialize_ising_model(y, num_states, mask)
    
    # Proceso ICM (Beta bajo para no emborronar demasiado)
    x_final = apply_ising_model_icm(y, x_init, params, num_states, beta=0.1, max_iterations=5, mask=mask)

    # --- VISUALIZACIÓN ---
    # Limpiamos el exterior para que se vea negro total (0)
    x_vis_init = x_init.copy()
    x_vis_final = x_final.copy()
    x_vis_init[~mask] = 0
    x_vis_final[~mask] = 0
    
    factor = 255 // (num_states - 1)

    cv.namedWindow('Original', cv.WINDOW_NORMAL)
    cv.imshow('Original', y.astype(np.uint8))
    
    cv.namedWindow('K-means Inicial', cv.WINDOW_NORMAL)
    cv.imshow('K-means Inicial', (x_vis_init * factor).astype(np.uint8))
    
    cv.namedWindow('ICM Final', cv.WINDOW_NORMAL)
    cv.imshow('ICM Final', (x_vis_final * factor).astype(np.uint8))
    
    print("Pulsa cualquier tecla en las ventanas de imagen para cerrar.")
    cv.waitKey(0)
    cv.destroyAllWindows()