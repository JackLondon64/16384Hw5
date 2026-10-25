"""Write HW5's fixtures: code/tests/inK.txt (case inputs) and outK.txt (answers).

    python -m solutions.gen_ref_solution_data code

One JSON object per line. The inputs are what `local_autograder.py` feeds the
student's code and what Gradescope perturbs; the outputs come from the
reference in this directory.

    in1.txt  total_energy   a batch of states per line
    in2.txt  dynamics       one state per line
    in3.txt  simulate       one initial state per line, T = 1 s, dt = 0.01 s
    in4.txt  plane          four touched corners, plus points to carry each
                            way between the paper and the robot (Q2, lab/plane.py)

Also checks the reference against MuJoCo's own double pendulum, when MuJoCo is
importable: see `check_mujoco`.
"""

import json
import math
import os
import random
import sys

import numpy as np

# The reference modules import each other flat, because that is how they sit
# in the Gradescope zip. Put this directory on the path so they resolve here
# too, however this is run.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))

import local_autograder as local  # noqa: E402
import ref_double_pendulum as ref  # noqa: E402
import ref_plane  # noqa: E402
from ref_answers import solve_dynamics, solve_energy, solve_plane, solve_simulate  # noqa: E402

rng = random.Random(384)  # 16-384: fixed seed for reproducible fixtures

# The parameter sets the previous expected.npz used; kept so the handout's
# cases carry over. Default first.
FIXED_PARAMS = [
    dict(M1=1.0, M2=1.0, L1=1.0, L2=1.0, G=9.81),
    dict(M1=2.0, M2=0.5, L1=1.0, L2=1.0, G=9.81),
    dict(M1=1.0, M2=1.0, L1=0.6, L2=1.3, G=9.81),
    dict(M1=0.8, M2=1.7, L1=1.2, L2=0.7, G=3.71),
]

SIM_T = 1.0
SIM_DT = 0.01

# How much of the simulate tolerance a fixed-step Euler integrator may use on
# a kept case. Well under 1, so Gradescope's jitter can't push it over.
EULER_MARGIN = 0.6
# The handout's original cases are kept at a slightly looser margin.
EULER_MARGIN_FIXED = 0.8


def _round(x, digits=3):
    return round(x, digits)


def _params():
    """A random pendulum: masses 0.3-3 kg, rods 0.4-1.6 m, gravity 1.6-25 m/s^2."""
    return dict(
        M1=_round(rng.uniform(0.3, 3.0)),
        M2=_round(rng.uniform(0.3, 3.0)),
        L1=_round(rng.uniform(0.4, 1.6)),
        L2=_round(rng.uniform(0.4, 1.6)),
        G=_round(rng.choice([9.81, 9.81, 3.71, 1.62, 24.79, rng.uniform(1.6, 25.0)])),
    )


def _state(max_speed=5.0):
    q = [rng.uniform(-math.pi, math.pi) for _ in range(2)]
    qd = [rng.uniform(-max_speed, max_speed) for _ in range(2)]
    return q, qd


def _write(path_in, path_out, cases, solve):
    with open(path_in, "w") as fin, open(path_out, "w") as fout:
        for case in cases:
            fin.write(json.dumps(case) + "\n")
            fout.write(json.dumps(solve(case)) + "\n")


def gen_energy(path_in, path_out):
    cases = []
    for params in FIXED_PARAMS + [_params() for _ in range(16)]:
        states = [_state() for _ in range(20)]
        cases.append(dict(params=params, q=[s[0] for s in states], qd=[s[1] for s in states]))
    _write(path_in, path_out, cases, solve_energy)
    return len(cases)


def gen_dynamics(path_in, path_out):
    cases = []
    # Rest and the two unstable equilibria, where qdd is zero: a wrong answer
    # shows up there as a sign or a missing term with nothing to hide behind.
    for q in ([0.0, 0.0], [math.pi, math.pi], [0.0, math.pi]):
        cases.append(dict(params=FIXED_PARAMS[0], q=q, qd=[0.0, 0.0]))
    for params in FIXED_PARAMS:
        for _ in range(15):
            q, qd = _state()
            cases.append(dict(params=params, q=q, qd=qd))
    for _ in range(25):
        params = _params()
        for _ in range(6):
            q, qd = _state()
            cases.append(dict(params=params, q=q, qd=qd))
    _write(path_in, path_out, cases, solve_dynamics)
    return len(cases)


def euler_usage(case):
    """The largest fraction of the simulate tolerance that explicit or
    semi-implicit Euler at the case's dt uses up, compared with the reference."""
    args = (case["params"], np.array(case["q0"]), np.array(case["qd0"]), case["T"], case["dt"])
    _, q, qd = ref.simulate(*args)
    worst = 0.0
    for semi in (False, True):
        _, eq, eqd = ref.euler(*args, semi_implicit=semi)
        worst = max(worst, np.abs(eq - q).max() / local.SIM_Q_TOL,
                    np.abs(eqd - qd).max() / local.SIM_QD_TOL)
    return worst


def discriminates(case):
    """A case where standing still fails: the pendulum has to really move."""
    _, q, _ = ref.simulate(case["params"], np.array(case["q0"]), np.array(case["qd0"]),
                           case["T"], case["dt"])
    return np.abs(q - q[0]).max() > 4 * local.SIM_Q_TOL


def gen_simulate(path_in, path_out):
    sim = dict(T=SIM_T, dt=SIM_DT)
    cases = [  # the previous expected.npz's cases
        dict(params=FIXED_PARAMS[0], q0=[0.3, -0.2], qd0=[0.0, 0.0], **sim),
        dict(params=FIXED_PARAMS[0], q0=[2.0, 1.0], qd0=[0.0, 0.0], **sim),
        dict(params=FIXED_PARAMS[0], q0=[0.0, 0.0], qd0=[3.0, 0.0], **sim),
        dict(params=FIXED_PARAMS[0], q0=[-0.8, 1.2], qd0=[1.0, -2.0], **sim),
        dict(params=FIXED_PARAMS[3], q0=[1.2, -0.4], qd0=[0.5, 0.0], **sim),
    ]
    for case in cases:
        usage = euler_usage(case)
        assert usage <= EULER_MARGIN_FIXED, f"Euler uses {usage:.2f} of the tolerance on {case}"

    drawn = rejected = 0
    while len(cases) < 30:
        params = rng.choice(FIXED_PARAMS) if len(cases) < 15 else _params()
        q0, qd0 = _state(max_speed=3.0)
        case = dict(params=params, q0=q0, qd0=qd0, **sim)
        drawn += 1
        if euler_usage(case) <= EULER_MARGIN and discriminates(case):
            cases.append(case)
        else:
            rejected += 1
    print(f"  simulate: kept {drawn - rejected} of {drawn} random cases "
          f"(the rest were too hard for Euler, or barely moved)")
    _write(path_in, path_out, cases, solve_simulate)
    return len(cases)


def _rotation_to(normal):
    """A rotation taking +z to `normal`, about the axis perpendicular to both."""
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    axis = np.cross([0.0, 0.0, 1.0], normal)
    s, c = np.linalg.norm(axis), normal[2]
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def plane_case(width, height, center, yaw, normal, noise):
    """Four touched corners of a sheet, and points to carry across.

    The sheet's x axis is turned `yaw` about its normal; every touch lands
    within `noise` (m, per axis) of its corner, so the four are not coplanar.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    R = _rotation_to(normal) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], float)
    local -= [width / 2, height / 2, 0]
    corners = local @ R.T + center
    corners += np.array([[rng.uniform(-noise, noise) for _ in range(3)] for _ in range(4)])
    H = ref_plane.plane_frame(corners)
    paper = [[rng.uniform(0, width), rng.uniform(0, height), rng.uniform(-0.01, 0.05)]
             for _ in range(6)]
    robot = [list(np.asarray(center) + [rng.uniform(-0.2, 0.2) for _ in range(3)])
             for _ in range(6)]
    return dict(corners=corners.tolist(), H=H.tolist(), paper_points=paper, robot_points=robot)


def gen_plane(path_in, path_out):
    cases = []
    # The simulator's own sheet, as touched exactly and as touched by hand.
    sys.path.insert(0, os.path.join(HERE, "..", "lab"))
    import drawing_arm as da

    exact = da.sim_paper_corners()
    for noise in (0.0, 0.002):
        corners = exact + np.array([[rng.uniform(-noise, noise) for _ in range(3)]
                                    for _ in range(4)])
        H = ref_plane.plane_frame(corners)
        cases.append(dict(corners=corners.tolist(), H=H.tolist(),
                          paper_points=[[0.05, 0.05, 0.0], [0.2, 0.1, -0.001], [0.1, 0.15, 0.03]],
                          robot_points=[[0.42, 0.0, 0.0], [0.3, -0.1, 0.05], [0.5, 0.1, 0.01]]))
    # Square to the robot, flat, as on the lab table: x along the robot's +y.
    cases.append(plane_case(0.297, 0.210, [0.42, 0.0, 0.0], math.pi / 2, [0, 0, 1], 0.0))
    # Sheets on the table, a little tilted and turned, touched by hand.
    for _ in range(25):
        tilt, heading = rng.uniform(0, math.radians(15)), rng.uniform(-math.pi, math.pi)
        normal = [math.sin(tilt) * math.cos(heading), math.sin(tilt) * math.sin(heading),
                  math.cos(tilt)]
        cases.append(plane_case(rng.uniform(0.15, 0.6), rng.uniform(0.1, 0.45),
                                [rng.uniform(0.25, 0.55), rng.uniform(-0.2, 0.2),
                                 rng.uniform(-0.05, 0.3)],
                                rng.uniform(-math.pi, math.pi), normal,
                                rng.uniform(0.0, 0.003)))
    # Any orientation at all, whiteboards on walls included: the formula
    # doesn't care which way is up.
    for _ in range(10):
        normal = [rng.gauss(0, 1) for _ in range(3)]
        cases.append(plane_case(rng.uniform(0.15, 0.6), rng.uniform(0.1, 0.45),
                                [rng.uniform(0.2, 0.6), rng.uniform(-0.3, 0.3),
                                 rng.uniform(0.0, 0.5)],
                                rng.uniform(-math.pi, math.pi), normal,
                                rng.uniform(0.0, 0.003)))
    _write(path_in, path_out, cases, solve_plane)
    return len(cases)


MJCF = """
<mujoco>
  <option gravity="0 0 -{G}" integrator="RK4" timestep="1e-4"/>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 -{L1}" mass="{M1}" diaginertia="1e-12 1e-12 1e-12"/>
      <body name="link2" pos="0 0 -{L1}">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 -{L2}" mass="{M2}" diaginertia="1e-12 1e-12 1e-12"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def check_mujoco():
    """Simulate the simulate fixtures' pendulums in MuJoCo and compare.

    MuJoCo's joints are relative, and a hinge about +y turns +z towards +x,
    which in the handout's (x, y-up) plane is counterclockwise from hanging
    down, so theta_1 = j1 and theta_2 = j1 + j2. Prints the worst difference
    over every case's 1 s; RK4 at 1e-4 s puts it around 1e-6 rad.
    """
    try:
        import mujoco
    except ImportError:
        print("  MuJoCo not importable, reference check skipped")
        return

    worst = 0.0
    root = os.path.join(HERE, "..", "code", "tests")
    with open(os.path.join(root, "in3.txt")) as fin, open(os.path.join(root, "out3.txt")) as fout:
        for line_in, line_out in zip(fin, fout):
            case, expected = json.loads(line_in), json.loads(line_out)
            model = mujoco.MjModel.from_xml_string(MJCF.format(**case["params"]))
            data = mujoco.MjData(model)
            q0, qd0 = case["q0"], case["qd0"]
            data.qpos[:] = [q0[0], q0[1] - q0[0]]
            data.qvel[:] = [qd0[0], qd0[1] - qd0[0]]
            steps = int(round(case["dt"] / model.opt.timestep))
            for k, want in enumerate(expected["q"]):
                if k:
                    mujoco.mj_step(model, data, nstep=steps)
                q = [data.qpos[0], data.qpos[0] + data.qpos[1]]
                worst = max(worst, float(np.max(np.abs(np.subtract(q, want)))))
    print(f"  reference vs MuJoCo over the simulate cases: worst |q error| {worst:.2e} rad")
    if worst > 1e-4:
        raise SystemExit("The reference and MuJoCo disagree: check ref_double_pendulum.py.")


if __name__ == "__main__":
    root = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "code", "tests")
    os.makedirs(root, exist_ok=True)
    counts = [
        ("in1.txt", gen_energy(f"{root}/in1.txt", f"{root}/out1.txt")),
        ("in2.txt", gen_dynamics(f"{root}/in2.txt", f"{root}/out2.txt")),
        ("in3.txt", gen_simulate(f"{root}/in3.txt", f"{root}/out3.txt")),
        ("in4.txt", gen_plane(f"{root}/in4.txt", f"{root}/out4.txt")),
    ]
    for name, count in counts:
        print(f"  tests/{name}: {count} cases (and its out file)")
    check_mujoco()
