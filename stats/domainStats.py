from __future__ import annotations
import math
import numpy as np
from skimage.measure import regionprops

def computeArea(domain)->float:
    return float(domain.area)

def computePerimeter(domain)->float:
    return float(domain.perimeter)

def computeRoughness(domain)->float:
    area=computeArea(domain)
    perimeter=computePerimeter(domain)
    if area>0 and perimeter>0:
        return perimeter**2/(4*math.pi*area)
    return 1.5

"dict struct:"
"state 1:"
"   Domain 1:"
"       area: X"
"       Perimeter: X"
"       Roughness: X"

def computeDomainStats(state_images:dict[int,np.ndarray])->dict[int,dict[int,dict[str,float]]]:
    result: dict[int, dict[int, dict[str, float]]] = {}
    for state, labeled in state_images.items():
        state_stats: dict[int, dict[str, float]] = {}
        for domain in regionprops(labeled):
            metrics: dict[str, float] = {
                "area":      computeArea(domain),
                "perimeter": computePerimeter(domain),
                "roughness": computeRoughness(domain),
            }
            state_stats[int(domain.label)] = metrics
        result[state]=state_stats
        
    return result
            