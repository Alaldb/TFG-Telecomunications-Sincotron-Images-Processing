#Crea clase procesador que reciba un objeto de la clase Directory
from matplotlib import image
import sklearn.cluster as cluster
from load import Directory
import cv2, numpy as np, tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns

class ImgProcessor:
    def __init__(self, directory_obj, parameters=None):
        if not isinstance(directory_obj, Directory):
            raise ValueError("El objeto debe ser una instancia de la clase Directory.")
        self.directory = directory_obj
        self.paths = self.directory.img_array
        self.method = self.directory.method
        self.parameters = parameters
        self.results = []

    def process_images(self):
        for path in self.paths:
            img=self.standarize_image(path)
            mask=self.create_circular_mask(img)
            if self.method == "thresholding":
                result = self.thresholding(img, self.parameters['thresholding'], mask)
            elif self.method == "ising":
                x_init, params = self.initialize_ising_model(img, num_states=self.parameters['ising']['num_states'], mask=mask)
                result = self.apply_ising_model_icm(img, x_init, params, num_states=self.parameters['ising']['num_states'], beta=self.parameters['ising']['beta'], max_iterations=self.parameters['ising']['max_iterations'])
            else:
                raise ValueError(f"Método no soportado: {self.method}")
            self.results.append({"original_img": img, "parameters": self.parameters, "result": result})

    #Standarization of images to 8 bits grayscale, with different methods for png and tiff

    def preprocess_images(self):
        processed = []
        for path in self.paths:
            ext = path.suffix.lower()
            match ext:
                case ".png":
                    standard = self.standarize_png(path)
                case ".tif" | ".tiff":
                    standard = self.standarize_tiff(path)
                case _:
                    raise ValueError(f"Formato no soportado: {ext}")
            processed.append(standard)
        return processed
    
    def standarize_tiff(self,path):
        img = tiff.imread(path)
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        standard = img8 if len(img8.shape) == 2 else cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        return standard
    
    def standarize_png(self,path):
        img = cv2.imread(path)
        standard  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return standard

    #Circular mask creation
    def create_circular_mask(self,image):
        # Convertir a 8 bits para HoughCircles
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.medianBlur(image_8bit, 5)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=int(image.shape[0]*0.3), maxRadius=int(image.shape[0]*0.6)
        )
        # Inicializar máscara como toda la imagen (True) o vacía (False) según si se detecta el círculo
        mask = np.zeros(image.shape, dtype=bool)
        # Si se detecta un círculo, dibujarlo en la máscara
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x_c, y_c, r = circles[0, 0]
            
            mask_img = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask_img, (x_c, y_c), r, 255, -1)
            # Convertir a booleano (True donde es 255)
            mask = mask_img == 255
        else:
            # Si no se detecta el círculo, usar toda la imagen como máscara (True)
            mask = np.ones(image.shape, dtype=bool)
            
        return mask
    
    #Thresholding methods: clahe, gaussian blur and otsu thresholding
    def apply_clahe(self, img, parameters, mask=None):
        clahe = cv2.createCLAHE(
            clipLimit=parameters['clip_limit'],
            tileGridSize=parameters['title_grid_size']
        )

        if mask is None:
            return clahe.apply(img)


        img_mod = img.copy()
        #al hacer que los píxeles fuera de la máscara tomen el valor mediano de la región útil, evitamos que el contraste se vea afectado por valores extremos fuera de la región de interés
        img_mod[~mask] = np.median(img[mask])
        clahe_full = clahe.apply(img_mod)

        result = img.copy()
        # Solo actualizamos los píxeles dentro de la máscara con el resultado de CLAHE, dejando el resto sin cambios
        result[mask] = clahe_full[mask]

        return result
    
    def apply_gaussian_blur(self, img, parameters):
        smoothed_img=cv2.GaussianBlur(img,parameters['ksize'],parameters['sigma'])
        return smoothed_img
    
    def apply_otsu_threshold(self, img):
        _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_img
    
    def thresholding(self, img, parameters, mask=None):
        #apply clahe, gaussian blur and otsu thresholding
        clahe_img = self.apply_clahe(img, parameters['clahe'], mask)
        gaussian_img = self.apply_gaussian_blur(clahe_img, parameters['gaussian_blur'])
        thresh_img = self.apply_otsu_threshold(gaussian_img)
        return thresh_img
    
    #Ising method
    def calculate_statistical_variables(self,y, x, num_states,mask=None):
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
    
    def initialize_ising_model(self, y, num_states, mask=None):
        if mask is not None:
            y_kmeans = y[mask].reshape(-1, 1) #lo hacemos compatible con kmeans para que no confunda valores con features
        else:
            y_kmeans = y.reshape(-1, 1)
        kmeans=cluster.KMeans(n_clusters=num_states, n_init=10, random_state=64)#para asegurar mas consistencia 10 iteraciones y una semilla fija para que sea reproducible (64 porque me gustan las potencias de 2)
        x_labels=kmeans.fit_predict(y_kmeans)
        x_matrix = np.full(y.shape, -1, dtype=np.int32)
        if mask is not None:
            x_matrix[mask] = x_labels
        else:
            x_matrix = x_labels.reshape(y.shape)
        parameters=self.calculate_statistical_variables(y,x_matrix,num_states,mask)
        return x_matrix, parameters
    
    def calculate_energy(self,y_value, x, mu, sigma, beta, state, row, col):
        #parte estadistica
        positive_energy=0.5*((y_value-mu)/sigma)**2
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
    
    def apply_ising_model_icm(self,y, x, parameters, num_states, beta, max_iterations):
        mask=self.create_circular_mask(y)
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
                        energy = self.calculate_energy(y[row, col], x_iteration, parameters[state]['mean'], parameters[state]['std'], beta, state, row, col)
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
            parameters = self.calculate_statistical_variables(y, x_iteration, num_states, mask)
        x_final = x_iteration.copy()
        return x_final
