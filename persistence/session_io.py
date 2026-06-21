import io
import json
import zipfile
import numpy as np
from core.session import Session
from core.segmentationContainer import SegmentationMethod
from openpyxl import Workbook

def saveSession(session:Session, path:str)->None:
    json_data={
        "image_name": session.image_name,
        "timestamp": session.timestamp,
        "parameters": session.parameters,
        "ising_stats": session.ising_stats,
        "domain_stats": keysToStr(session.domain_stats),
        "segmentation_method": session.segmentation_method.value
    }
    arrays={}
    if session.original_image is not None:
        arrays["original_image"] = session.original_image
    if session.corrected_image is not None:
        arrays["corrected_image"] = session.corrected_image
    if session.ising_result is not None:
        arrays["ising_result"] = session.ising_result
    
    for state, labeled_image in session.domain_data.get("labeled_images", {}).items():
        arrays[f"labeled_images_{state}"] = labeled_image
    
    buffer=io.BytesIO()#Creamos un buffer en memoria para escribir el zip
    np.savez(buffer, **arrays)#Guardamos los arrays en el buffer el **arrays es para pasar cada array como un argumento separado
    buffer.seek(0)#Volvemos al inicio del buffer para leerlo después
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:#Creamos el zip
        zip_file.writestr("session.json", json.dumps(json_data))#Guardamos el json en el zip
        zip_file.writestr("arrays.npz", buffer.read())#Guardamos el buffer con los arrays en el zip


def keysToStr(dictionary:dict)->dict:#JSON cannot serialize keys that are not strings, so we convert them to strings before saving and back to ints when loading
    if isinstance(dictionary, dict):
        return {str(key): keysToStr(value) for key, value in dictionary.items()}#recursively convert keys to strings
    elif isinstance(dictionary, list):
        return [keysToStr(item) for item in dictionary]
    else:
        return dictionary


def loadSession(path:str)->Session:
    with zipfile.ZipFile(path, "r") as zip_file:
        json_data=json.loads(zip_file.read("session.json").decode("utf-8"))#Leemos el json del zip
        npz_data=io.BytesIO(zip_file.read("arrays.npz"))#Leemos el npz del zip
    arrys=np.load(npz_data)#Cargamos los arrays del npz
    labeled_images={}
    for key in arrys.files:
        if key.startswith("labeled_images_"):
            state=int(key.split("_")[-1])#Extraemos el estado del nombre del array
            labeled_images[state]=arrys[key]
    method_str = json_data.get("segmentation_method")
    segmentation_method = SegmentationMethod(method_str) if method_str is not None else SegmentationMethod.ICM #compatibilidad con sesiones anteriores
    session=Session(
        image_name=json_data["image_name"],
        original_image=arrys["original_image"] if "original_image" in arrys else None,
        corrected_image=arrys["corrected_image"] if "corrected_image" in arrys else None,
        ising_result=arrys["ising_result"] if "ising_result" in arrys else None,
        domain_data={"labeled_images": labeled_images},
        parameters=json_data.get("parameters", {}),
        ising_stats=json_data.get("ising_stats", {}),
        domain_stats=strKeysToInt(json_data.get("domain_stats", {})),
        timestamp=json_data.get("timestamp", ""),
        segmentation_method=segmentation_method
    )
    return session

def strKeysToInt(dictionary:dict)->dict:#Convertimos las claves de nuevo a enteros al cargar
    if isinstance(dictionary, dict):
        result={}
        for key, value in dictionary.items():
            new_key = int(key) if key.lstrip("-").isdigit() else key
            result[new_key] = strKeysToInt(value)
        return result
    elif isinstance(dictionary, list):
        return [strKeysToInt(item) for item in dictionary]
    else:
        return dictionary

def exportCorrectedImage(session:Session, path:str)->None:
    if session.corrected_image is None:
        raise ValueError("Session has no corrected image to export.")
    import cv2
    cv2.imwrite(path, session.corrected_image)

def exportDataExcel(session: Session, path:str, min_area:float=0)->None:
    file=Workbook()
    file.remove(file.active)
    for state, domains in session.domain_stats.items():
        current_page=file.create_sheet(title=f"Sate {state+1}")
        filtered_domains={domain_id:metrics
                          for domain_id, metrics in domains.items() if metrics["area"]>=min_area}
        
        if filtered_domains:
            metric_names=[domain_metrics.upper() for domain_metrics in next(iter(filtered_domains.values())).keys()]
            current_page.append(["DOMAIN ID"]+metric_names)
            for domian_id, metrics in filtered_domains.items():
                current_page.append([domian_id]+[metrics[metric.lower()] for metric in metric_names])
    file.save(path)