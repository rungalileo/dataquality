from enum import Enum
from typing import Dict, List

import numpy as np
from pydantic import BaseModel


class ErrorType(str, Enum):
    classification = "classification"
    undetected = "undetected"


class Pixel(BaseModel):
    x: int
    y: int

    def deserialize(self) -> List[List[int]]:
        return [[self.x, self.y]]


class Contour(BaseModel):
    pixels: List[Pixel]


class Polygon(BaseModel):
    # id: int
    # label_idx: int
    # lablel: str
    # error_type: ErrorTypes
    contours: List[Contour]

    def deserialize(self) -> List[np.ndarray]:
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.deserialize())
            contours.append(np.array(pixels))
        return contours

    def deserialize_json(self) -> List[List[List[List[List[int]]]]]:
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.deserialize())
            contours.append([pixels])
        return contours


class PolygonMap(BaseModel):
    map: Dict[int, List[Polygon]]
