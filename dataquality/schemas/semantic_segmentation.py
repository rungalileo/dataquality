from enum import Enum
from typing import List

import numpy as np
from pydantic import BaseModel


class SemSegCols(str, Enum):
    id = "id"
    image_path = "image_path"
    mask_path = "mask_path"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"


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
    contours: List[Contour]

    def deserialize(self) -> List[np.ndarray]:
        """Takes a polygon object and returns a list of np.ndarrays
        corresponding to the contours of the polygon
        """
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.deserialize())
            contours.append(np.array(pixels))
        return contours

    def deserialize_json(self) -> List[List[List[List[List[int]]]]]:
        """Takes a polygon object and returns a list of lists etc
        of contours of the polygon for json consumption
        """
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.deserialize())
            contours.append([pixels])
        return contours
