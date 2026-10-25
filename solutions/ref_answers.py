"""The expected answer for one case of each HW5 test, from the reference.

Shared by `gen_ref_solution_data.py`, which writes these out as
code/tests/outK.txt, and by Gradescope's `grade.py`, which calls them on
perturbed inputs. Each takes a case exactly as it appears in an inK.txt line
and returns the dict that goes on the matching outK.txt line.

Imports are flat (`ref_double_pendulum`, not `solutions.ref_double_pendulum`):
in the Gradescope zip these files all sit side by side at the root.
"""

import numpy as np

import ref_double_pendulum as ref
import ref_plane


def _tolist(value):
    return np.asarray(value, dtype=float).tolist()


def solve_energy(case):
    return {"energy": _tolist(ref.total_energy(case["params"], case["q"], case["qd"]))}


def solve_dynamics(case):
    return {"qdd": _tolist(ref.dynamics(case["params"], case["q"], case["qd"]))}


def solve_simulate(case):
    t, q, qd = ref.simulate(case["params"], np.array(case["q0"]), np.array(case["qd0"]),
                            case["T"], case["dt"])
    return {"t": _tolist(t), "q": _tolist(q), "qd": _tolist(qd)}


def solve_plane(case):
    """The frame from the case's corners; the point transforms on the case's own H,
    so a wrong plane_frame isn't charged to them too."""
    H = np.array(case["H"], dtype=float)
    return {
        "H": _tolist(ref_plane.plane_frame(case["corners"])),
        "robot_points": _tolist(ref_plane.paper_to_robot(H, case["paper_points"])),
        "paper_points": _tolist(ref_plane.robot_to_paper(H, case["robot_points"])),
    }
