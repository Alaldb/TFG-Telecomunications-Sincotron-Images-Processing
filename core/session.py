from __future__ import annotations #para permitir que python no se queje con mas de un tipo 
from dataclasses import dataclass, field
from  datetime import datetime
import numpy as np

@dataclass #Permite incializadores y funciones básicas automáticas
class Session:
    image_name: str
    original_image: np.ndarray
    corrected_image: np.ndarray | None = None
    ising_result: np.ndarray | None = None
    domain_data: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    temporal_ising_object: object | None = field(default=None, repr=False) #Aquí guardamos el objeto de Ising para poder usarlo en toda la pipeline
