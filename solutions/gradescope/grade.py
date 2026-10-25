"""Gradescope harness for HW5.

The difference from `local_autograder.py` is where the expected answers come
from. Locally they are read out of tests/outK.txt, which ships with the
assignment, so a student could in principle read the answers off disk. Here
nothing is read: every case's inputs are jittered by a fresh random amount on
every run, and the expected output is computed from the reference solution at
those perturbed inputs. Two submissions never see the same numbers, and there
is nothing to hard-code against.

The tests themselves are `local_autograder.py`'s `grade_*` functions, imported
rather than reimplemented, so a case passes or fails here for the same reasons
it does locally.

Layout Gradescope gives us:

    /autograder/source/       this file and everything build.py put in the zip
    /autograder/submission/   the student's unpacked submission.zip
                              (double_pendulum.py, lab/plane.py, ...)
    /autograder/results/      results.json goes here
"""

import json
import os
import random
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import local_autograder as local  # noqa: E402
import ref_plane  # noqa: E402
from ref_answers import solve_dynamics, solve_energy, solve_plane, solve_simulate  # noqa: E402

SUBMISSION = Path(os.environ.get("GS_SUBMISSION", "/autograder/submission"))
RESULTS = Path(os.environ.get("GS_RESULTS", "/autograder/results")) / "results.json"

# Most cases each test is graded on: a random sample of the fixture when it
# has more, which is a second reason two submissions don't see the same test.
SAMPLE = int(os.environ.get("GS_SAMPLE", "400"))

MAX_FAILURES_SHOWN = 8

# Jitter. Small enough that every case stays the problem it was (the simulate
# cases were chosen with room for an Euler integrator, and this keeps them
# inside it), large enough that no answer computed for the published fixtures
# survives.
ANGLE_JITTER = 0.05  # rad, absolute
RELATIVE_JITTER = 0.02  # masses, lengths, gravity, velocities
SPEED_JITTER = 0.05  # rad/s, absolute, so a zero velocity moves too

CORNER_JITTER = 0.002  # m, each coordinate of each touched corner
POINT_JITTER = 0.005  # m, the points carried between paper and robot

POINTS = {
    "dynamics": 20,
    "simulate": 15,
    "energy": 5,
    "plane_frame": 6,
    "paper_to_robot": 2,
    "robot_to_paper": 2,
}


# ----------------------------------------------------------------------
# Finding the student's code
# ----------------------------------------------------------------------


def find_submission_file(name):
    """The student's `name`, wherever in the submission they put it (the
    shallowest one, in case the archiver added a wrapper directory)."""
    found = list(SUBMISSION.rglob(name))
    return min(found, key=lambda p: len(p.parts)) if found else None


def load_student(name):
    """(module, error). Exactly one of the two is None."""
    path = find_submission_file(name)
    if path is None:
        return None, f"{name} not found in the submission"
    try:
        return local._load_module(path), None
    except Exception:
        return None, f"{name} failed to import:\n{traceback.format_exc()}"


# ----------------------------------------------------------------------
# Cases: read, sample, perturb
# ----------------------------------------------------------------------


def sample_cases(path, rng, count=SAMPLE):
    """At most `count` cases from a JSONL fixture, by reservoir sampling."""
    reservoir = []
    with open(path) as handle:
        for i, line in enumerate(line for line in handle if line.strip()):
            if len(reservoir) < count:
                reservoir.append(json.loads(line))
            else:
                j = rng.randrange(i + 1)
                if j < count:
                    reservoir[j] = json.loads(line)
    return reservoir


def _angle(rng, value):
    return float(value) + rng.uniform(-ANGLE_JITTER, ANGLE_JITTER)


def _scale(rng, value):
    return float(value) * (1.0 + rng.uniform(-RELATIVE_JITTER, RELATIVE_JITTER))


def _speed(rng, value):
    return _scale(rng, value) + rng.uniform(-SPEED_JITTER, SPEED_JITTER)


def _params(rng, params):
    return {name: _scale(rng, value) for name, value in params.items()}


def _states(rng, rows, jitter):
    return [[jitter(rng, v) for v in row] for row in rows]


def perturb_energy(case, rng):
    return dict(params=_params(rng, case["params"]),
                q=_states(rng, case["q"], _angle),
                qd=_states(rng, case["qd"], _speed))


def perturb_dynamics(case, rng):
    return dict(params=_params(rng, case["params"]),
                q=[_angle(rng, v) for v in case["q"]],
                qd=[_speed(rng, v) for v in case["qd"]])


def perturb_simulate(case, rng):
    return dict(params=_params(rng, case["params"]),
                q0=[_angle(rng, v) for v in case["q0"]],
                qd0=[_speed(rng, v) for v in case["qd0"]],
                T=case["T"], dt=case["dt"])


def perturb_plane(case, rng):
    corners = [[float(v) + rng.uniform(-CORNER_JITTER, CORNER_JITTER) for v in c]
               for c in case["corners"]]
    jitter = lambda rows: [[float(v) + rng.uniform(-POINT_JITTER, POINT_JITTER)  # noqa: E731
                            for v in row] for row in rows]
    # H is an input to the point tests; recomputed so it stays a proper frame.
    return dict(corners=corners, H=ref_plane.plane_frame(corners).tolist(),
                paper_points=jitter(case["paper_points"]),
                robot_points=jitter(case["robot_points"]))


def cases(fixture, perturb, solve, rng):
    """(case, expected) pairs: sampled, perturbed, and answered by the reference."""
    for case in sample_cases(HERE / fixture, rng):
        case = perturb(case, rng)
        yield case, solve(case)


# ----------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------


def _failed(name, reason):
    grader = local.Grader(name, keep=MAX_FAILURES_SHOWN)
    grader.record("submitted", False, reason)
    return grader


def grade_all(rng):
    """[(key, Grader)] for every autograded test, in POINTS order."""
    keep = MAX_FAILURES_SHOWN
    tests = {
        "dynamics": ("double_pendulum.py", "dynamics", local.grade_dynamics,
                     "tests/in2.txt", perturb_dynamics, solve_dynamics),
        "simulate": ("double_pendulum.py", "simulate", local.grade_simulate,
                     "tests/in3.txt", perturb_simulate, solve_simulate),
        "energy": ("double_pendulum.py", "total_energy", local.grade_energy,
                   "tests/in1.txt", perturb_energy, solve_energy),
        "plane_frame": ("plane.py", "plane_frame", local.grade_plane_frame,
                        "tests/in4.txt", perturb_plane, solve_plane),
        "paper_to_robot": ("plane.py", "paper_to_robot", local.grade_paper_to_robot,
                           "tests/in4.txt", perturb_plane, solve_plane),
        "robot_to_paper": ("plane.py", "robot_to_paper", local.grade_robot_to_paper,
                           "tests/in4.txt", perturb_plane, solve_plane),
    }
    modules = {}
    out = []
    for key in POINTS:
        filename, name, grade, fixture, perturb, solve = tests[key]
        if filename not in modules:
            modules[filename] = load_student(filename)
        module, error = modules[filename]
        if module is None:
            out.append((key, _failed(name, error)))
        elif not hasattr(module, name):
            out.append((key, _failed(name, f"{name} not found in {filename}")))
        else:
            out.append((key, grade(module, cases(fixture, perturb, solve, rng), keep)))
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def as_test(grader, max_score):
    """One Gradescope test, scored by the fraction of cases that passed."""
    fraction = (grader.passed / grader.total) if grader.total else 0.0
    lines = [f"{grader.passed}/{grader.total} cases passed."]
    if grader.failures:
        lines.append("")
        lines.append("First failures:")
        for label, detail in grader.failures:
            lines.append(f"  [FAIL] {label}" + (f" - {detail}" if detail else ""))
        remaining = (grader.total - grader.passed) - len(grader.failures)
        if remaining > 0 and not grader.stopped:
            lines.append(f"  ... and {remaining} more.")
    return {
        "name": grader.name,
        "score": round(fraction * max_score, 3),
        "max_score": max_score,
        "output": "\n".join(lines),
    }


def main():
    # A fresh seed per run: this is what makes the perturbation unguessable.
    rng = random.Random(int.from_bytes(os.urandom(8), "big"))

    tests = [as_test(grader, POINTS[key]) for key, grader in grade_all(rng)]

    results = {
        "tests": tests,
        "output": (
            "Cases are the published fixtures with small random perturbations "
            "(to the angles, velocities, masses, lengths and gravity, and to the "
            "touched corners and points of the plane cases) applied "
            "fresh on every run, checked against the reference solution. "
            "Otherwise your marks should match what local_autograder.py tells "
            "you. If they don't, the difference is code that only works for "
            "the exact published numbers."
        ),
        "visibility": "visible",
        "stdout_visibility": "hidden",
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(results, handle, indent=2)

    total = sum(t["score"] for t in tests)
    possible = sum(t["max_score"] for t in tests)
    print(f"wrote {RESULTS}: {total:g}/{possible}")
    for test in tests:
        print(f"  {test['name']}: {test['score']:g}/{test['max_score']}")


if __name__ == "__main__":
    main()
