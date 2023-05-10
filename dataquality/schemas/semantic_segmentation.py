from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, StrictInt


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


class Community(BaseModel):
    """Information about the classes in a community for class overlap

    Each community comes with a score and a list of relevant classes

    A community represents a group of labels that the model finds to be very similar and
    is confusing between. A community is at the labels level, not at sample level.
    Communities might look like this:
    communities = [
        {"score": 0.53, "labels": ["a", "c"], "num_samples": 24},
        {"score": 0.22, "labels": ["b", "e", "f", "g"], "num_samples": 101},
    ]

    The labels are the labels in this particular community. They do not overlap across
    communities.
    `num_samples` is exactly the total number of samples for all of these classes. There
    is nothing done here at the sample level, so this is just the sum of the counts
    The score is determined as a probability mass of the non-GT class for that community
    for each sample, averaged. That is to say, for each sample of a community, you
    take the sum of the classes that are not the GT, but are in the community. You
    then average all of those across the samples.
    Ex:
        Sample 1, GT = 4, community = [1, 3, 4]
            prob vector: [0.1, 0.05, 0.05, 0.25, 0.55]
            Probability mass/sum = 0.05+0.25 = 0.3
        Sample 2, GT = 1, community = [1, 3, 4]
            prob vector: [0.1, 0.6, 0.1, 0.1, 0.1]
            Probability mass/sum = 0.1+0.1 = 0.2
        Score = (0.3 + 0.2) / 2 = 0.25

        So the score for this community would be 0.25, and the num_samples would be 2
    """

    score: float
    labels: List[int]
    num_samples: StrictInt
