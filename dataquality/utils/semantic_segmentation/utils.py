import cv2
import numpy as np


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """
    Convert binary mask to boundary mask. The boundary mask is a mask tracing the
    edges of the object in the mask. Therefore, the inside of the mask is now hollow
    which means our IoU calculation will only be on the 'boundary' of each polygon.
    The dilation ratio controls how large the tracing is around the polygon edges.
    The smaller the dilation ratio the smaller the boundary mask will be and thus
    likely decrease Boundary IoU.

    This function creates an eroded mask which essentially shrinks all the polygons
    and then subtracts the eroded mask from the original mask to get the boundary mask.

    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation
        dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    if mask.shape[1] == 1:
        mask = mask.squeeze(1)
    mask = mask.astype(np.uint8)
    n, h, w = mask.shape
    for im in range(n):
        img_diag = np.sqrt(h**2 + w**2)
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        # Pad image so mask truncated by the image border is also considered as boundary
        new_mask = cv2.copyMakeBorder(
            mask[im], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        # if the number does not equal either the old mask or 0 then set to 0
        mask_erode = np.where(mask_erode != mask[im], 0, mask_erode)
        boundary_mask = mask[im] - mask_erode
        # G_d intersects G in the paper.
        mask[im] = boundary_mask

    return mask
