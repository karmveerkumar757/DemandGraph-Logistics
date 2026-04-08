from abc import ABC, abstractmethod


class FleetAllocationModule(ABC):
    @abstractmethod
    def allocate(self, *args, **kwargs) -> dict:
        """Reserve capacity for future fleet assignment logic."""


class EtaPredictionModule(ABC):
    @abstractmethod
    def predict_eta(self, *args, **kwargs) -> dict:
        """Reserve capacity for future delivery ETA prediction logic."""

