from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from priority_map.modules.Segment import Segmentation

"""
Module for object localization using different provided metadata

Localization Context can be expanded to include more information as needed, 
such as camera intrinsics, GPS data, etc.

And a respective child class of Object Localizer can be implemented 
to use that information for localization.
"""

@dataclass
class LocalizationContext:
    frame: object
    image: np.ndarray
    curr_pos: tuple[float, float, float]
    flow_transform: tuple[float, float] | None = None

class ObjectLocalizer(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def localize(self, segmentation: Segmentation, context: LocalizationContext):
        """Return segmentation with geo_pos filled."""
        pass