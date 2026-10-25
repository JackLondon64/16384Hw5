# Gradescope autograder — HW5

Instructor-only. This lives under `solutions/`, which is gitignored, because the
zip it builds contains the answer key.

## Build and upload

From the assignment root (the folder with `assignment5.tex` and `code/`):

```bash
python -m solutions.gen_ref_solution_data code   # only if the fixtures are stale
python solutions/gradescope/build.py             # -> solutions/gradescope/autograder.zip
```

Upload `autograder.zip` under **Configure Autograder** on the Gradescope
assignment. Nothing else needs setting up there.

The generator needs numpy and scipy. When MuJoCo is importable (the course's
`16384` conda env has it), it also simulates every `simulate` case with
MuJoCo's own point-mass double pendulum and prints the worst disagreement with
the reference, which is about 1e-10 rad. That is an independent check of the
reference's equations and sign conventions; it exits with an error if the two
disagree by more than 1e-4.

## Marks

| test | points | writeup part |
|---|---:|---|
| `dynamics` | 20 | Q1 (a) |
| `simulate` | 15 | Q1 (b) |
| `total_energy` | 5 | Q1 (c) |
| `plane_frame` | 6 | Q2 (a), `lab/plane.py` |
| `paper_to_robot` | 2 | Q2 (a) |
| `robot_to_paper` | 2 | Q2 (a) |

That is 50 autograded points. The other 40 are Q2 (c), graded by hand from the
video in `lab/handin/`; there is no PDF. Q1 totals 40 and Q2 50, 90 in all.
Q2 (b), the simulation run, and Q1's chaos experiment are not graded.
Each test scores
the fraction of cases that passed, so partial credit is automatic. Change the
split in `POINTS` in `grade.py`.

## Why the cases move

`local_autograder.py` compares against `code/tests/outK.txt`, which ships with
the assignment, so the answers are on the student's disk. This grader never
reads them. On every run it jitters the case *inputs* and computes the answers
from the reference solution at those perturbed values:

- angles by ±0.05 rad
- masses, lengths, gravity and velocities by ±2%, and velocities by a further
  ±0.05 rad/s, so a zero velocity moves too
- the plane cases' touched corners by ±2 mm per coordinate (their `H` is then
  recomputed from the reference, since it is an input to the point tests), and
  the points carried across by ±5 mm

Two submissions therefore never see the same numbers, and a lookup table built
from the published fixtures scores zero. Jittering the parameters also means
every Gradescope case uses a non-default pendulum, so code that writes `9.81`
or `1.0` where it should use `G` or `M1` fails across the board here, even
though it passes the default-parameter cases locally.

The tests are `local_autograder.py`'s `grade_*` functions, imported, so a case
passes or fails for the same reasons on both.

## Tolerances

- **`total_energy` and `dynamics`** are closed form: 1e-6, relative to the
  answer's size once it is above 1. Any correct rearrangement agrees to
  rounding.
- **`simulate`** is compared at 0.25 rad and 1.5 rad/s per sample over 1 s,
  with `t` exact to 1e-9. That is loose on purpose: the handout invites
  students to write their own integrator, and a fixed-step Euler (explicit or
  semi-implicit) at dt = 0.01 should pass.
- **The simulate cases are filtered so that stays true.** The generator keeps
  a random case only if both Euler variants use at most 60% of either
  tolerance on it, and only if the pendulum moves more than 1 rad, so a
  `simulate` that returns the initial state every step fails. The five cases
  carried over from the old `expected.npz` are kept at up to 80%. 25 runs of
  this grader on each Euler variant all scored full marks.

The bugs this was checked against, and what the local grader says for each: a
sign error in a coupling term (fails nearly every dynamics and simulate case),
a hard-coded `G` (fails every non-default case), a wrong potential-energy zero
(fails `total_energy` only), and `np.linspace(0, T, T / dt)` for the times
(fails `simulate` on shape). `solve_ivp` with its default tolerances passes.

## What's in the zip

```
grade.py                the harness
run_autograder          entry point; sets MPLBACKEND=Agg first
setup.sh                pip install -r requirements.txt
requirements.txt        numpy, scipy, matplotlib
local_autograder.py     copied from code/; grade.py imports its tests
ref_double_pendulum.py  the answer key (Q1)
ref_plane.py            the answer key (Q2's lab/plane.py)
ref_answers.py          one "expected answer" function per test, shared with
                        gen_ref_solution_data.py
tests/inK.txt           case inputs only, no outK.txt
```

`double_pendulum.py` and `plane.py` are looked for anywhere in the submission
(the shallowest match), so a wrapper directory added by the archiver doesn't
matter.

The plane tests have a fixed formula to match (the diagonals' cross product,
the plane through the centroid), stated in the stub and the writeup, so the
answer is unique and compared at 1e-6. The point tests are handed the case's
own `H`, so a wrong `plane_frame` isn't charged to them too.

## Testing a change

```bash
S=$(mktemp -d); mkdir -p $S/source $S/submission $S/results
unzip -q solutions/gradescope/autograder.zip -d $S/source
(cd $S/submission && unzip -q /path/to/submission.zip)

MPLBACKEND=Agg GS_SUBMISSION=$S/submission GS_RESULTS=$S/results \
  python3 $S/source/grade.py
```

A complete, correct solution should print 50/50 on every run.
`solutions/ref_submission/` is one, laid out like the handout: `code/double_pendulum.py`
(the three TODOs filled in the way the handout asks, `solve_ivp` at
`RTOL`/`ATOL`) and `lab/` (`plane.py` filled in, A4's reference `robot.py`, and
a `strokes/rkd.json`). Its `code/submission.zip` is the upload. If you edit it,
rebuild the zip with
`cd solutions/ref_submission/code && python ../../../code/create_submission.py`.
The shipped stubs score 0/50, with one "not implemented yet" line per pendulum
test and no traceback.

To try the reference through the lab in the simulator, copy `lab/` somewhere,
drop in `ref_submission/lab/*`, and run `check_plane.py`, `calibrate.py` and
`draw.py strokes/rkd.json` there.
