from matplotlib import colors, pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.mixture import GaussianMixture
import cv2
import gco
import tifffile
from core.segmentationContainer import SegmentationContainer,SegmentationMethod

class GraphCutsService:
    def __init__(self,  num_states: int=3, 
                 lambda_value: float=1.0, 
                 sigma: float|None=None, 
                 num_iterations: int=-1, 
                 number_gaussians_per_state: int=3):
        self.num_states=num_states
        self.lambda_value=lambda_value
        self.sigma=sigma
        self.num_iterations=num_iterations
        self.number_gaussians_per_state=number_gaussians_per_state
    
    def run(self, image:np.ndarray)->None:
        self.original_image=image
        self.mask=self.create_mask()
        self.initial_labels=self.initialize_model()
        self.final_image=self.apply_graph_cuts()
        self.parameters=self.calculate_parameters()
    
    def search_circles(self):
        img = cv2.normalize(self.original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)#type: ignore
        blurred = cv2.medianBlur(img, 5)

        height = self.original_image.shape[0]
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=height, #Only 1 circle
            param1=100,
            param2=100,
            minRadius=int(height * 0.4),
            maxRadius=int(height * 0.55),
        )

        if circles is not None:
            circles = [(x, y, r) for x, y, r in np.round(circles[0]).astype(int)]
            return circles
        else:
            print("No se detecto ningun circulo.")

    def create_mask(self):
        circles=self.search_circles()
        if circles is None:
            return np.ones(self.original_image.shape, dtype=bool)

        mask = np.zeros(self.original_image.shape, dtype=np.uint8)
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, 1, thickness=-1)
        return mask.astype(bool)
    
    def initialize_model(self):
        original_image_for_Kmeans = self.original_image[self.mask].reshape(-1, 1)
        kmeans = cluster.KMeans(n_clusters=self.num_states, n_init=10, random_state=64)
        clustered_image = kmeans.fit_predict(original_image_for_Kmeans)
        masked_clustered_image = np.full(self.original_image.shape, -1, dtype=np.int32)
        masked_clustered_image[self.mask] = clustered_image
        return masked_clustered_image
    
    def create_gaussian_mixtures_for_states(self, labels: np.ndarray)->dict:
        gaussian_mixtures={}
        for state in range(self.num_states):
            pixel_values=self.original_image[(labels==state) & self.mask].reshape(-1,1)#Esto se hace porque gaussian mixture solo es capaz de recibir arrays 2D
            k=min(self.number_gaussians_per_state,len(pixel_values))#Evita que el parámetro metido por el usuario provoque errores
            k=min(k,len(np.unique(pixel_values)))#Evita más gaussianas que valores únicos
            gaussian_mixture=GaussianMixture(n_components=k, covariance_type="full",random_state=64)
            gaussian_mixture.fit(pixel_values)
            gaussian_mixtures[state]=gaussian_mixture
        return gaussian_mixtures
    
    #esta funcion es equivalente al primer sumatorio que indica la probabilidad de un pixel de pertenecer a una etiqueta
    def compute_data_cost(self, gaussian_mixtures: dict)->np.ndarray:
        height,width=self.original_image.shape
        pixel_intensities = self.original_image.astype(np.float64).reshape(-1, 1)
        data_cost=np.zeros((height*width, self.num_states), dtype=np.float64)# esto crea una matriz que tiene el mismo numero de filas que H*W y una columna por estado esto es necesario para que podamos usar score_samples
        for state, gaussian_mixture in gaussian_mixtures.items():
            data_cost[:,state]=-gaussian_mixture.score_samples(pixel_intensities)#este score samples da los valores en negativo, a mas diferente más negativo por eso le ponemos el -
        max_val = data_cost[self.mask.ravel()].max()
        if max_val > 0:
            data_cost /= max_val  # normaliza a [0, 1]
        data_cost[~self.mask.ravel()]=0.0#Con esto truncamos todos los valores, aunque de la sensación que estamos poniendoles costes bajos, como los vamos a desconectar en los N-Links realmente no importa
        data_cost-=data_cost[self.mask.ravel()].min() #Con esto estamos haciendo que el menor valor sea 0 y reducir lo máximo posible el valor máximo, aunque el valor más parecido tendrá el mismo valor que los valores externos, como esos valores no están conectados por N-Links no influyen
        return data_cost.reshape(height,width,self.num_states)#Con esto lo devolvemos a la forma original de manera que sea usable en el pipeline
    
    def estimate_sigma(self)->float: #Este método básicamente calcula una rms de los píxeles vecinos, rms es útil porque con rms estamos calculando la dispersión respecto a 0
        vertical_diff=np.diff(self.original_image.astype(np.float64),axis=0).ravel()
        horizontal_diff=np.diff(self.original_image.astype(np.float64),axis=1).ravel()
        diffs=np.concatenate([vertical_diff,horizontal_diff])#Con esto hemos conseguido una matriz 1D que representa todas las diferencias entre los píxeles vecinos con conexión 4
        rms=float(np.sqrt(np.mean(diffs**2)))+1e-6#Con el término 1e-6 evitamos en el hipotético caso que sea 0 que sea justamente 0
        return rms
    
    #Esto es el segundo sumatorio la biblioteca que uso necesita los pesos verticales y horizontales separados para poderse calcular
    def compute_nlink_weights(self)->tuple[np.ndarray,np.ndarray]:
        sigma=self.sigma if self.sigma is not None else self.estimate_sigma()
        img=self.original_image.astype(np.float64)

        #1º Se calculan las diferencias de los vecionos
        vertical_diff=img[1:,:]-img[:-1,:]
        horizontal_diff=img[:,1:]-img[:,:-1]

        # Vij = lambda * exp(-(Ii - Ij)² / 2σ²)
        vertical_weights=self.lambda_value*np.exp(-vertical_diff**2/(2*sigma**2))
        horizontal_weights=self.lambda_value*np.exp(-horizontal_diff**2/(2*sigma**2))

        #Desconectar píxeles en el borde al hacerlos pasar a 0 no tiene coste cortar la arista y el algoritmo los ignorará
        vertical_weights*=(self.mask[1:,:]&self.mask[:-1,:]).astype(np.float64)
        horizontal_weights*=(self.mask[:,1:]&self.mask[:,:-1]).astype(np.float64)
        return vertical_weights.astype(np.float64),horizontal_weights.astype(np.float64)

    def apply_graph_cuts(self)->np.ndarray:
        gaussian_mixtures=self.create_gaussian_mixtures_for_states(self.initial_labels)
        data_cost=self.compute_data_cost(gaussian_mixtures)
        vertical_weights,horizontal_weights=self.compute_nlink_weights()
        #Este paso es algo lioso, secuerda submodularidad que si tienen etiquetas iguales tiene que penalizar menos que si son diferentes
        #Para conseguir esto vamos a dar la vuelta a una matriz identidad de forma que si las etiquetas son iguales no penalice para mas info busca sobre Potts
        #|0,1,1|
        #|1,0,1|
        #|1,1,0|
        pairwise_cost=(1-np.eye(self.num_states)).astype(np.float64)
        result=gco.cut_grid_graph(
            unary_cost=data_cost,
            pairwise_cost=pairwise_cost,
            cost_v=vertical_weights,
            cost_h=horizontal_weights,
            n_iter=self.num_iterations,
            algorithm='expansion'
        )
        final=result.reshape(self.original_image.shape).astype(np.int32)
        final[~self.mask]=-1
        return final
    

    #con este metodo homogeneizamos con ICM y volvemos el cálculo de los estados una caja negra
    def calculate_parameters(self)->dict:
        parameters={}
        for state in range(self.num_states):
            pixel_intensities=self.original_image[(self.final_image==state)&self.mask]
            if len(pixel_intensities)>0:
                parameters[state]={
                    'mean': float(np.mean(pixel_intensities)),
                    'std': float(np.std(pixel_intensities))+1e-6
                }
            else:
                parameters[state]={'mean':0,'std':0}
        return parameters
    
    def getSegmentationContainer(self)->SegmentationContainer:
        return SegmentationContainer(
            original_image=self.original_image,
            mask=self.mask,
            final_image=self.final_image,
            num_states=self.num_states,
            parameters=self.parameters,
            initial_labels=self.initial_labels,
            method=SegmentationMethod.GRAPH_CUTS,
            method_configuration={
                "lambda_value":               self.lambda_value,
                "sigma":                      self.sigma,
                "num_iterations":             self.num_iterations,
                "number_gaussians_per_state": self.number_gaussians_per_state
            }
        )
    
    

IMAGE_PATH = r"C:\Users\user\Desktop\034_XMCD_Ni_120rot_FOV20_+-500mA__corrected.tif"
#r"C:\Users\user\Desktop\009_XMCD_Ni_corrected.tif"

if __name__ == "__main__":
    image = tifffile.imread(IMAGE_PATH)

    gc = GraphCutsService(
        num_states=3,
        lambda_value=0.06,
        sigma=None,
        num_iterations=-1,
        number_gaussians_per_state=5
    )

    gc.run(image)

    n = gc.num_states
    base_colors = ["black"] + list(colors.TABLEAU_COLORS.values())[:n]
    cmap = colors.ListedColormap(base_colors[:n + 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    im1 = axes[1].imshow(gc.initial_labels, cmap=cmap, vmin=-1, vmax=n - 1)
    axes[1].set_title("KMeans inicial")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], ticks=range(-1, n))

    im2 = axes[2].imshow(gc.final_image, cmap=cmap, vmin=-1, vmax=n - 1)
    axes[2].set_title("Graph Cuts final")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], ticks=range(-1, n))

    plt.tight_layout()
    plt.show()