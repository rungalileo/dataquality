from enum import Enum
from typing import Dict, List, Optional

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
    none = None


class Pixel(BaseModel):
    x: int
    y: int

    def deserialize_opencv(self) -> List[List[int]]:
        """Takes a pixel object and returns JSON compatible list

        We deserialize to a JSON compatible format that matches what OpenCV
        expects when drawing contours.

        OpenCV expects a list of list of pixel coordinates.
        """
        return [[self.x, self.y]]

    def deserialize_json(self) -> List[int]:
        """Takes a pixel object and returns JSON compatible list"""
        return [self.x, self.y]


class Contour(BaseModel):
    pixels: List[Pixel]


class Polygon(BaseModel):
    id: int
    label_idx: int
    misclassified_class_label: Optional[int] = None
    error_type: ErrorType = ErrorType.none
    contours: List[Contour]

    def deserialize_opencv(self) -> List[np.ndarray]:
        """Deserialize the contours in a polygon to be OpenCV contour compatible

        OpenCV.drawContours expects a list of np.ndarrays corresponding
        to the contours in the polygon.

        Example:
            polygon = Polygon(
                contours=[Contour(pixels=[Pixel(x=0, y=0), Pixel(x=0, y=1)])]
            )
            polygon.deserialize_opencv()
            >>> [np.array([[0, 0], [0, 1]])]
        """
        contours = []
        for contour in self.contours:
            pixels = [pixel.deserialize_opencv() for pixel in contour.pixels]
            contours.append(np.array(pixels))
        return contours

    def deserialize_json(self) -> Dict:
        """Deserialize a polygon object to be JSON compatible for Minio upload

        We export polygons as a nested list of pixels for the Frontend to draw. This
        nested list format is required by the OpenCV library to draw contours.

        Example:
            polygon = Polygon(
                contours=[Contour(pixels=[Pixel(x=0, y=0), Pixel(x=0, y=1)])]
            )
            polygon.deserialize_json()
            >>> [[[[[0, 0]], [[0, 1]]]]]
        """
        contours = []
        for contour in self.contours:
            pixels = [pixel.deserialize_json() for pixel in contour.pixels]
            contours.append(pixels)

        return {
            "id": self.id,
            "label_idx": self.label_idx,
            "error_type": self.error_type.value,
            "polygon": contours,
        }
