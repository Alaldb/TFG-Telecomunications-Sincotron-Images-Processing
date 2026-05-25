import subprocess, time, os, cv2, numpy as np
from tkinter import Tk, filedialog
class ImageJ:
    def __init__ (self, fiji_path=None):
        self.folder_path = './Elements'
        self.fiji_path = fiji_path if fiji_path else self.search_fiji()
        self.image_path = self.search_fiji_image_path()
        self.temporal_image_path = self.normalize_image_for_fiji()
        pass

    def search_fiji_image_path(self):
        root = Tk()
        root.withdraw()  # Only the file explorer
        root.attributes('-topmost', True)  # Open on top of other windows
        image_path= filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.tif;*.tiff;*.jpg;*.jpeg;*.png")])
        if image_path:
            self.image_path=os.path.normpath(image_path)#Correct path format for Windows
            return self.image_path
        else:
            raise ValueError("No image file selected. Please select an image to proceed.")


    def search_fiji(self):
        possible_names = ['fiji-windows-x64.exe', 'ImageJ-win64.exe', 'ImageJ.exe', 'fiji.exe']
        path = None
        print("Buscando Fiji en el disco C (omitiendo carpetas del sistema)...")
        for root, dirs, files in os.walk("C:\\"):
            if any(folder in root for folder in ["Windows", "$Recycle.Bin", "ProgramData", "AppData"]):
                continue
            for name in possible_names:
                if name in files:
                    path = os.path.join(root, name)
                    print(f"Fiji found: {path}")
                    break
            if path: 
                break
        if not path:
            raise ValueError('Fiji not found. Please install Fiji and try again.')
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        txt_path = os.path.join(self.folder_path, 'fiji_path.txt')
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.write(path)

        return path
    
    def normalize_image_for_fiji(self):
        img=cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if img.dtype == np.float32:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        temporal_img_path = os.path.join(self.folder_path, 'temporal_image.tif')
        cv2.imwrite(temporal_img_path, img)
        return temporal_img_path
    
    def modify_image_with_fiji(self):
        last_mod_time = os.path.getmtime(self.temporal_image_path)
        print('Open Fiji')
        fiji=subprocess.Popen([self.fiji_path, self.temporal_image_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
        while True:
            if fiji.poll() is not None: #if fiji.poll() is not None, it means that the process has finished (Fiji is closed)
                break
            current_mod_time = os.path.getmtime(self.temporal_image_path)
            if current_mod_time > last_mod_time:
                print('Changes detected, loading modified image...')
                time.sleep(0.5) #wait for the file to be fully written
                modified_img = cv2.imread(self.temporal_image_path, cv2.IMREAD_UNCHANGED)
                temporal_img_path = os.path.join(self.folder_path, 'temporal_image_modified.tif')
                cv2.imwrite(temporal_img_path, modified_img)
                self.temporal_image_path = temporal_img_path
                break









if __name__ == "__main__":
    print("=== TEST DE EJECUCIÓN DIRECTA EN SISTEMA ===")
    
    # Instanciamos tu clase
    herramienta = ImageJ()
    
    try:
        # Lanzamos la función para que barra tu disco real
        ruta_detectada = herramienta.search_fiji()
        
        print("\n=== RESULTADOS DEL TEST ===")
        print(f"La función devolvió: {ruta_detectada}")
        print(f"¿Existe el archivo devuelto?: {os.path.exists(ruta_detectada)}")
        
        # Comprobar si el archivo TXT se escribió bien
        archivo_txt = './Elements/fiji_path.txt'
        if os.path.exists(archivo_txt):
            with open(archivo_txt, 'r', encoding='utf-8') as f:
                contenido = f.read()
            print(f"Contenido del TXT generado: {contenido}")
            
    except ValueError as e:
        print("\n=== TEST FALLIDO ===")
        print(f"La función lanzó la excepción esperada cuando no encuentra nada:\n{e}")
        print("Nota: Si tienes Fiji instalado, asegúrate de que no esté en una carpeta oculta dentro de AppData o Windows.")
        
    except Exception as e:
        print("\n=== ERROR INESPERADO ===")
        print(f"El script falló por un problema imprevisto: {e}")

    print("=== PROBANDO EL SELECTOR DE IMÁGENES EN TU CLASE ===")
    
    # 1. Instanciamos la clase que estás desarrollando para tu TFG
    #herramienta = ImageJ()
    
    try:
        # 2. Llamamos a tu función para que despliegue el explorador de Windows
        print("[PROCESO] Abriendo el explorador de archivos...")
        ruta_retornada = herramienta.search_fiji_image_path()
        
        # 3. Verificaciones de éxito si el usuario seleccionó un archivo
        print("\n=== ¡TEST COMPLETADO CON ÉXITO! ===")
        print(f"-> Ruta devuelta por el método (return): {ruta_retornada}")
        print(f"-> Ruta almacenada en el objeto (self): {herramienta.image_path}")
        print(f"-> ¿El archivo existe realmente en el disco?: {os.path.exists(ruta_retornada)}")
        
    except ValueError as e:
        # Aquí capturamos la excepción que configuraste si el usuario cierra la ventana sin elegir nada
        print("\n=== CONTROL DE EXCEPCIÓN CORRECTO ===")
        print(f"La función lanzó el ValueError esperado:\n[Excepción]: {e}")
        
    except Exception as e:
        # Por si ocurre algún otro fallo imprevisto (como falta de imports de tkinter)
        print(f"\n[ERROR INESPERADO]: {e}")
    
    print("=== PROBANDO LA NORMALIZACIÓN DE IMAGEN ===")

    # Reutilizamos herramienta si ya tiene una imagen seleccionada,
    # o creamos una nueva instancia si no existe
    #try:
    #    ruta_imagen = herramienta.image_path
    #except (NameError, AttributeError):
    #    print("No hay instancia previa válida, creando una nueva...")
    #    herramienta = ImageJ()

    try:
        print(f"[INFO] Imagen de entrada: {herramienta.image_path}")
        
        # Comprobamos el dtype de la imagen antes de normalizar
        img_original = cv2.imread(herramienta.image_path, cv2.IMREAD_UNCHANGED)
        print(f"[INFO] Dtype de la imagen original: {img_original.dtype}")
        print(f"[INFO] Shape de la imagen original: {img_original.shape}")

        # Llamamos a la función a testear
        ruta_temporal = herramienta.normalize_image_for_fiji()

        # Verificaciones
        print(f"[INFO] Ruta del archivo temporal generado: {ruta_temporal}")
        print(f"[INFO] ¿El archivo temporal existe?: {os.path.exists(ruta_temporal)}")

        # Comprobamos el dtype de la imagen resultante
        img_normalizada = cv2.imread(ruta_temporal, cv2.IMREAD_UNCHANGED)
        print(f"[INFO] Dtype de la imagen normalizada: {img_normalizada.dtype}")
        print(f"[INFO] Shape de la imagen normalizada: {img_normalizada.shape}")
        print(f"[INFO] Valor mínimo: {img_normalizada.min()} | Valor máximo: {img_normalizada.max()}")

        print("\n=== NORMALIZACIÓN COMPLETADA CON ÉXITO ===")

    except Exception as e:
        print(f"\n[ERROR INESPERADO EN NORMALIZACIÓN]: {e}")

    print("=== PROBANDO APERTURA DE IMAGEN EN FIJI ===")

    try:
        print(f"[INFO] Ruta de Fiji: {herramienta.fiji_path}")
        print(f"[INFO] Imagen temporal a abrir: {herramienta.temporal_image_path}")

        # Verificaciones previas
        if not os.path.exists(herramienta.fiji_path):
            raise FileNotFoundError(f"No se encontró el ejecutable de Fiji en: {herramienta.fiji_path}")
        if not os.path.exists(herramienta.temporal_image_path):
            raise FileNotFoundError(f"No se encontró la imagen temporal en: {herramienta.temporal_image_path}")

        # Lanzamos Fiji con la imagen temporal ya generada en el __init__
        print("[PROCESO] Abriendo Fiji...")
        proceso = subprocess.Popen([herramienta.fiji_path, herramienta.temporal_image_path])
        print(f"[INFO] Fiji lanzado con PID: {proceso.pid}")
        print("\n=== FIJI ABIERTO CON ÉXITO ===")

    except FileNotFoundError as e:
        print(f"\n[ERROR - ARCHIVO NO ENCONTRADO]: {e}")

    except Exception as e:
        print(f"\n[ERROR INESPERADO]: {e}")