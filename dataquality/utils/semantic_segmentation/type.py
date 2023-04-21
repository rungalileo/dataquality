from typing import Dict, List

import numpy as np
from pydantic import BaseModel


class Pixel(BaseModel):
    coord: List[List[int]]


class Contour(BaseModel):
    pixels: List[Pixel]


class Polygon(BaseModel):
    contours: List[Contour]

    def unserialize(self):
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.coord)
            contours.append(np.array(pixels))
        return contours

    def unserialize_json(self):
        contours = []
        for contour in self.contours:
            pixels = []
            for pixel in contour.pixels:
                pixels.append(pixel.coord)
            contours.append([pixels])
        return contours


class Polygon_Map(BaseModel):
    map: Dict[int, List[Polygon]]

    def unserialize(self):
        unserialized_map = {}
        for key, blobs in self.map.items():
            unserialized_blobs = []
            for blob in blobs:
                unserialized_blobs.append(blob.unserialize())
            unserialized_map[key] = unserialized_blobs
        return unserialized_map

    def unserialize_json(self):
        unserialized_map = []
        counter = 0
        for key, polygons in self.map.items():
            for polygon in polygons:
                polygon = polygon.unserialize_json()
                polygon_object = {'id': counter, 
                                  'label_int': key, 
                                  'error_type': 'none',
                                  'polygon': polygon}
                counter += 1
                unserialized_map.append(polygon_object)
        return unserialized_map
