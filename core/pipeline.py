from __future__ import annotations
import numpy as np
from stats.domainStats import computeDomainStats
from core.session import Session
from processing.corrector import Corrector
from processing.isingMethodService import Ising
from processing.domainService import DomainService
from core.segmentationContainer import SegmentationContainer,SegmentationMethod

class PipelineDictator:

    def __init__(self, coverage: float = 0.80, beta: float = 2, 
                 max_iterations: int =20, num_states: int =3):
        self.coverage = coverage
        self.beta = beta
        self.max_iterations = max_iterations
        self.num_states =  num_states
    
    def apply_correction(self, image: np.ndarray, image_name: str) -> Session:
        session = Session(
            image_name=image_name,
            original_image=image.copy(),
            parameters={
                "coverage": self.coverage,
                "beta": self.beta,
                "max_iterations": self.max_iterations,
                "num_states": self.num_states,
            }
        )
        corrector = Corrector(coverage=self.coverage)
        session.corrected_image = corrector.apply_correction(image=image)
        return session

    def run_ising(self, session: Session) -> Session:
        ising = Ising(
            beta=self.beta,
            max_iterations=self.max_iterations,
            num_states=self.num_states
        )
        ising.run(session.corrected_image)
        session.ising_result=ising.final_image
        session.segmentation_container = ising.getSegmentationContainer()
        session.segmentation_method = SegmentationMethod.ICM
        return session
    
    def run_domains(self, session: Session) -> Session:
        domain_service=DomainService(session.segmentation_container)
        session.domain_data=domain_service.get_data()
        session.domain_stats=computeDomainStats(session.domain_data["labeled_images"])
        return session