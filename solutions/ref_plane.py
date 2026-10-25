"""Reference for lab/plane.py: the paper frame from four touched corners.

`ref_submission/lab/plane.py` is the same thing written into the student stub.
"""

import numpy as np


def plane_frame(corners):
    p1, p2, p3, p4 = np.asarray(corners, dtype=float)
    n = np.cross(p3 - p1, p4 - p2)
    n /= np.linalg.norm(n)
    c = (p1 + p2 + p3 + p4) / 4.0
    o = p1 - np.dot(p1 - c, n) * n
    x = (p2 - p1) - np.dot(p2 - p1, n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    H = np.eye(4)
    H[:3, 0], H[:3, 1], H[:3, 2], H[:3, 3] = x, y, n, o
    return H


def paper_to_robot(H, points):
    points = np.asarray(points, dtype=float)
    return points @ H[:3, :3].T + H[:3, 3]


def robot_to_paper(H, points):
    points = np.asarray(points, dtype=float)
    return (points - H[:3, 3]) @ H[:3, :3]
