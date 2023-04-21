from pydantic import BaseModel
from typing import List, Dict
import numpy as np

class Pixel(BaseModel):
    coord: List[List[int]]

class Contour(BaseModel):
    pixels: List[Pixel]

class Blob(BaseModel):
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
    
class Contour_Map(BaseModel):
    map: Dict[int, List[Blob]]
    
    def unserialize(self):
        unserialized_map = {}
        for key, blobs in self.map.items():
            unserialized_blobs = []
            for blob in blobs:
                unserialized_blobs.append(blob.unserialize())
            unserialized_map[key] = unserialized_blobs
        return unserialized_map
    
    def unserialize_json(self):
        unserialized_map = {}
        for key, blobs in self.map.items():
            unserialized_blobs = []
            for blob in blobs:
                unserialized_blobs.append(blob.unserialize_json())
            unserialized_map[key] = unserialized_blobs
        return unserialized_map