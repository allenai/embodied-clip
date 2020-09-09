"""Calculates the IoU between 2 three-dimensional bounding boxes.

The bounding box calculations only rely on corner points of the box:
np.array([
   [1.37129939, 0.7017756, -1.469296],
   [1.0239253, 0.701771259, -1.77404511],
   [1.32596815, 0.7017716, -2.11833453],
   [1.67334223, 0.7017759, -1.81358528],
   [1.37129772, 0.9279571, -1.46929729],
   [1.02392352, 0.927952766, -1.77404642],
   [1.32596636, 0.927953064, -2.11833572],
   [1.67334056, 0.9279575, -1.81358659]
])

At a high level:
* Uniformly samples points between the 2 bounding boxes.
* For both boxes, check if the point is within the box.
* Sum the total number of points in both boxes, and divide that by the number
  of points in the union of both boxes.
"""
from typing import List, Tuple, Sequence
from scipy.spatial import ConvexHull, Delaunay
import numpy as np


def iou(b1: np.ndarray, b2: np.ndarray, num_points: int = 2197):
    """Calculates the IoU between 3d bounding boxes b1 and b2."""
    def _outer_bounds(points_1: np.ndarray, points_2: np.ndarray):
        """Sample points from the outer bounds formed by points_1 and points_2."""
        assert points_1.shape == points_2.shape
        bounds = dict()
        for i in range(len(points_1)):
            x1, y1, z1 = points_1[i]
            x2, y2, z2 = points_2[i]
            points = [(x1, 'x'), (x2, 'x'),
                      (y1, 'y'), (y2, 'y'),
                      (z1, 'z'), (z2, 'z')]
            for val, d_key in points:
                if d_key not in bounds:
                    bounds[d_key] = {'min': val, 'max': val}
                else:
                    if val > bounds[d_key]['max']:
                        bounds[d_key]['max'] = val
                    elif val < bounds[d_key]['min']:
                        bounds[d_key]['min'] = val
        return bounds

    def _in_box(box: np.ndarray, points: np.ndarray) -> Sequence[bool]:
        """For each point, return if its in the hull."""
        hull = ConvexHull(box)
        deln = Delaunay(box[hull.vertices])
        return deln.find_simplex(points) >= 0

    bounds = _outer_bounds(b1, b2)
    dim_points = int(num_points ** (1 / 3))

    xs = np.linspace(bounds['x']['min'], bounds['x']['max'], dim_points)
    ys = np.linspace(bounds['y']['min'], bounds['y']['max'], dim_points)
    zs = np.linspace(bounds['z']['min'], bounds['z']['max'], dim_points)
    points = np.array(
        [[x, y, z] for x in xs for y in ys for z in zs], copy=False)

    in_b1 = _in_box(b1, points)
    in_b2 = _in_box(b2, points)

    intersection = np.count_nonzero(in_b1 * in_b2)
    union = np.count_nonzero(in_b1 + in_b2)
    return intersection / union
