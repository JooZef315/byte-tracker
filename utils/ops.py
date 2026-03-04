import numpy as np

def xywh2ltwh(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyltwh format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def bbox_ioa(
    box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7
) -> np.ndarray:
    """
    Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (
        np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)
    ).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(
        0
    )

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def _get_covariance_matrix(boxes: np.ndarray):
    """
    Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (np.ndarray): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (tuple): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = np.concatenate((boxes[:, 2:4] ** 2 / 12, boxes[:, 4:5]), axis=-1)
    a, b, c = np.split(gbbs, 3, axis=-1)
    a, b, c = a.squeeze(-1), b.squeeze(-1), c.squeeze(-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = cos**2
    sin2 = sin**2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x1, y1 = x1.squeeze(-1), y1.squeeze(-1)
    x2, y2 = np.split(obb2[..., :2], 2, axis=-1)
    x2, y2 = x2.squeeze(-1)[None, :], y2.squeeze(-1)[None, :]

    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    a2, b2, c2 = a2[None, :], b2[None, :], c2[None, :]

    t1 = (
        (
            (a1[:, None] + a2) * (y1[:, None] - y2) ** 2
            + (b1[:, None] + b2) * (x1[:, None] - x2) ** 2
        )
        / ((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2 + eps)
    ) * 0.25
    t2 = (
        ((c1[:, None] + c2) * (x2 - x1[:, None]) * (y1[:, None] - y2))
        / ((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2 + eps)
    ) * 0.5
    t3 = (
        np.log(
            ((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2)
            / (
                4
                * (
                    np.maximum(a1[:, None] * b1[:, None] - c1[:, None] ** 2, 0)
                    * np.maximum(a2 * b2 - c2**2, 0)
                )
                ** 0.5
                + eps
            )
            + eps
        )
        * 0.5
    )
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd
