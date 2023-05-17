from enum import Enum
from typing import List, Optional

import numpy as np
from pydantic import BaseModel


class SemSegCols(str, Enum):
    id = "id"
    image = "image"
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

    @property
    def deserialize_opencv(self) -> List[List[int]]:
        """Takes a pixel object and returns JSON compatible list

        We deserialize to a JSON compatible format that matches what OpenCV
        expects when drawing contours.

        OpenCV expects a list of list of pixel coordinates.
        """
        return [[self.x, self.y]]

    @property
    def deserialize_json(self) -> List[int]:
        """Takes a pixel object and returns it as list of ints"""
        return [self.x, self.y]


class Contour(BaseModel):
    pixels: List[Pixel]


class Polygon(BaseModel):
    uuid: str  # UUID4
    label_idx: int
    misclassified_class_label: Optional[int] = None
    error_type: ErrorType = ErrorType.none
    contours: List[Contour]
    data_error_potential: Optional[float] = None

    @property
    def contours_opencv(self) -> List[np.ndarray]:
        """Deserialize the contours in a polygon to be OpenCV contour compatible

        OpenCV.drawContours expects a list of np.ndarrays corresponding
        to the contours in the polygon.

        Example:
            polygon = Polygon(
                contours=[Contour(pixels=[Pixel(x=0, y=0), Pixel(x=0, y=1)])]
            )
            polygon.contours_opencv
            >>> [np.array([[0, 0], [0, 1]])]
        """
        contours = []
        for contour in self.contours:
            pixels = [pixel.deserialize_opencv for pixel in contour.pixels]
            contours.append(np.array(pixels))
        return contours

    @property
    def contours_json(self) -> List:
        """Deserialize the contours as a JSON

        Example:
            polygon = Polygon(
                contours=[
                    Contour(pixels=[Pixel(x=0, y=0), Pixel(x=0, y=1)]),
                    Contour(pixels=[Pixel(x=12, y=9), Pixel(x=11, y=11)])
                ]
            )
            polygon.contours_opencv
            >>> [[[0, 0], [0, 1]], [[12, 9], [11, 11]]]
        """
        contours = []
        for contour in self.contours:
            pixels = [pixel.deserialize_json for pixel in contour.pixels]
            contours.append(pixels)
        return contours
