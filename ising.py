import numpy as np
import sklearn.cluster as cluster
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns


def upload_image():
    ruta = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
    
    try:
        y = tiff.imread(ruta)
        if len(y.shape) == 3:
            y = np.mean(y, axis=2)
        y = y.astype(np.float32)
        
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 255.0
        #aumenta contraste con ecualizacion de histograma
        #y = cv.equalizeHist(y.astype(np.uint8)).astype(np.float32)
        
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
        
        # 2. Dibujar el círculo en la máscara
        # Creamos una imagen negra y dibujamos un círculo blanco (255)
        mask_img = np.zeros(y.shape, dtype=np.uint8)
        cv.circle(mask_img, (x_c, y_c), r, 255, -1) # -1 para rellenar
        
        # Convertir a Booleano (True donde es 255)
        mask = mask_img == 255
        print(f"Máscara circular creada: Centro({x_c}, {y_c}), Radio({r})")
    else:
        print("No se detectó el círculo, se usará toda la imagen.")
        mask = np.ones(y.shape, dtype=bool) # Toda la imagen como True
        
    return mask

def calculate_statistical_variables(y, x, num_states,mask=None):
    parameters ={}
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

def initialize_ising_model(y, num_states):
    mask=get_circular_mask(y)
    y_kmeans = y[mask].reshape(-1, 1) #lo hacemos compatible con kmeans para que no confunda valores con features
    kmeans=cluster.KMeans(n_clusters=num_states, n_init=10, random_state=64)#para asegurar mas consistencia 10 iteraciones y una semilla fija para que sea reproducible (64 porque me gustan las potencias de 2)
    x_labels=kmeans.fit_predict(y_kmeans)
    x_matrix = np.full(y.shape, -1, dtype=np.int32)
    x_matrix[mask] = x_labels
    parameters=calculate_statistical_variables(y,x_matrix,num_states, mask)
    return x_matrix, parameters

def calculate_energy(y_value, x, mu, sigma, beta, state, row, col):
    #parte estadistica
    positive_energy=((y_value-mu)/sigma)**2
    #comparacion con vecinos
    neighbour_sum=0
    mov_tuples=[(-1,0),(1,0),(0,-1),(0,1)]
    rows, cols = x.shape
    for dx, dy in mov_tuples:
        neighbourx=row+dx
        neighboury=col+dy
        #verificar que el vecino este dentro de los limites de la imagen
        if 0 <= neighbourx < rows and 0 <= neighboury < cols:
            if x[neighbourx, neighboury] == state:
                neighbour_sum += 1
    negative_energy=-beta*neighbour_sum
    energy=positive_energy+negative_energy
    return energy


def apply_ising_model_icm(y, x, parameters, num_states, beta, max_iterations):
    mask=get_circular_mask(y)
    rows, columns=y.shape
    x_final = x.copy()
    x_iteration = x.copy()
    for iteration in range(max_iterations):
        x_old_iteration = x_iteration.copy()
        #recorrer filas
        for row in range(rows):
            #recorrer columnas
            for col in range(columns):
                if not mask[row, col]: continue # Si el pixel no está en la máscara, lo saltamos
                #inicializamos la energia del pixel a infinito
                best_energy = float('inf')
                best_state = x_iteration[row, col]
                #evaluamos cada estado posible
                for state in range(num_states):
                    #calcular la energia de ese estado
                    energy = calculate_energy(y[row, col], x_iteration, parameters[state]['mean'], parameters[state]['std'], beta, state, row, col)
                    #tomamos decision de cambiar o no estados
                    if energy < best_energy:
                        best_energy = energy
                        best_state = state
                #actualizamos el estado del pixel con la decision final
                x_iteration[row, col] = best_state
        #verificar si hubo cambios en la iteracion, si no hubo cambios se puede detener el proceso
        if np.array_equal(x_iteration, x_old_iteration) and iteration > 0:
            print(f"Converged at iteration {iteration}")
            x_final = x_iteration.copy()
            break
        #actualizamos los parametros estadisticos con el nuevo estado de la imagen
        parameters = calculate_statistical_variables(y, x_iteration, num_states, mask)
    x_final = x_iteration.copy()
    return x_final

# Ejemplo de ejecución
y = upload_image()
x_init, params = initialize_ising_model(y, num_states=3)
# Ejecutamos con beta=1.5 y 10 iteraciones máximo
x_final = apply_ising_model_icm(y, x_init, params, num_states=3, beta=2, max_iterations=10)

# Para visualizarlo con OpenCV
cv.imshow('Original', y.astype(np.uint8))
cv.namedWindow('K-means', cv.WINDOW_NORMAL) # Permite cambiar el tamaño
cv.imshow('K-means', (x_init * 120).astype(np.uint8)) # Escalado para ver tonos
cv.resizeWindow('K-means', 800, 800) # Ajusta a un tamaño cómodo
cv.namedWindow('Resultado ICM', cv.WINDOW_NORMAL) # Permite cambiar el tamaño
cv.imshow('Resultado ICM', (x_final * (255 // (3 - 1))).astype(np.uint8))#normalizamos para que se vea bien (255 dividido por el número de estados menos 1 para que el máximo sea 255)
cv.resizeWindow('Resultado ICM', 400, 400) # Ajusta a un tamaño cómodo
cv.namedWindow('ICM Final', cv.WINDOW_NORMAL) # Permite cambiar el tamaño
cv.imshow('ICM Final', (x_final * 120).astype(np.uint8))
cv.resizeWindow('ICM Final', 400, 400) # Ajusta a un tamaño cómodo
cv.waitKey(0)
                    
    

