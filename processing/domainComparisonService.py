from __future__ import annotations
import numpy as np

class DomainComparisonService:
    def __init__(self, domain_a: dict, domain_b: dict, image_shape: tuple[int,int])->None:
        self.domain_a=domain_a
        self.domain_b=domain_b
        self.image_diagonal=float(np.sqrt(image_shape[0]**2+image_shape[1]**2))
        
    def compare(self)->dict:
        return {
            "stat_diffs": self.computeStatDiffs(),
            "displacement": self.computeDisplacement(),
        }
    
    def computeStatDiffs(self)->dict:
        stats_a=self.domain_a["stats"]
        stats_b=self.domain_b["stats"]

        mean_intensity_a=float(np.mean(self.domain_a["values"])) if len(self.domain_a["values"]) else 0
        mean_intensity_b=float(np.mean(self.domain_b["values"])) if len(self.domain_b["values"]) else 0

        metrics={key: (stats_a[key], stats_b[key]) for key in stats_a}#Como ambos dominios tienen la misma forma de diccionario stats, iteramos sobre todas las keys de forma que el resultado de esta comparación sea dinámico y si cambian las stats no haya que hacer cambio alguno
        metrics["mean_intensity"]=(mean_intensity_a, mean_intensity_b)#Nueva entrada con la intensidad media de cada dominio

        result={}
        for name, (value_a, value_b) in metrics.items(): #Con el diccionario que hemos creado antes ahora podemos de forma cómoda iterar ya que las operaciones para sacar los resultados son las mismas.
            absolute=value_b-value_a
            percentage=(absolute/value_a*100) if value_a!=0 else 0
            result[name]={ #Aqui estamos creando un diccionario para cada métrica con todos los resultados incluyendo los valores iniciales que después representaremos
                "value_a": value_a,
                "value_b": value_b,
                "absolute": absolute,
                "percentage": percentage
            }
        return result
    
    def computeDisplacement(self)->dict:
        centroid_a=self.domain_a["coords"].mean(axis=0)#Con esto estamos calculando la media de la posición X y la posición Y
        centroid_b=self.domain_b["coords"].mean(axis=0)

        displ_vector_y=float(centroid_b[0]-centroid_a[0])#Recuerda que se guarda como [Y,X]
        displ_vector_x=float(centroid_b[1]-centroid_a[1])
        displ_vector_module=float(np.sqrt(displ_vector_x**2+displ_vector_y**2))
        distance_normalized=displ_vector_module/self.image_diagonal

        return{
            "centroid_a": (float(centroid_a[0]),float(centroid_a[1])),
            "centroid_b": (float(centroid_b[0]),float(centroid_b[1])),
            "displacement_vector":(displ_vector_x,displ_vector_y),
            "vector_module": displ_vector_module,
            "normalized_vector": distance_normalized
        }


