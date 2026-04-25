import cv2, numpy as np, tifffile as tiff


def create_circular_mask(image):
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

def apply_clahe(img,parameters,mask=None):
    clahe=cv2.createCLAHE(clipLimit=parameters['clip_limit'], tileGridSize=parameters['title_grid_size'])
    if mask is not None:
        clahe_img=clahe.apply(img[mask])
    else:
        clahe_img=img
    return clahe_img

def apply_gaussian_blur(img, parameters):
    smoothed_img=cv2.GaussianBlur(img,parameters['ksize'],parameters['sigma'])
    return smoothed_img

def apply_otsu_threshold(img):
    _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_img

def thresholding(img, parameters, mask=None):
    #apply clahe, gaussian blur and otsu thresholding
    clahe_img = apply_clahe(img, parameters['clahe'], mask)
    gaussian_img = apply_gaussian_blur(clahe_img, parameters['gaussian_blur'])
    thresh_img = apply_otsu_threshold(gaussian_img)
    return thresh_img

