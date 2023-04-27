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

    def unserialize(self) -> List[List[int]]:
        return [[self.x, self.y]]


class Contour(BaseModel):
    pixels: List[Pixel]


class Polygon(BaseModel):
    # id: int
    # label_idx: int
    # lablel: str
    # error_type: ErrorTypes
    contours: List[Contour]

    def unserialize(self):
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.unserialize())
            contours.append(np.array(pixels))
        return contours

    def unserialize_json(self):
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.unserialize())
            contours.append([pixels])
        return contours


class PolygonMap(BaseModel):
    """Maps a label to a list of polygons"""

    map: Dict[int, List[Polygon]]

    def unserialize(self):
        unserialized_map = {}
        for key, blobs in self.map.items():
            unserialized_blobs = []
            for blob in blobs:
                unserialized_blobs.append(blob.unserialize())
            unserialized_map[key] = unserialized_blobs
        return unserialized_map

    def unserialize_json(self) -> List[Dict]:
        unserialized_map = []
        counter = 0
        for lbl, polygons in self.map.items():
            for polygon in polygons:
                polygon = polygon.unserialize_json()
                polygon_object = {
                    "id": counter,
                    "label_idx": lbl,
                    "error_type": "none",
                    "polygon": polygon,
                }
                counter += 1
                unserialized_map.append(polygon_object)
        return unserialized_map
