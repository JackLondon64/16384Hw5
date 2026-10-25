"""Assemble autograder.zip for Gradescope.

Everything the container needs is copied in here rather than fetched at run
time, so the zip is the whole autograder:

    grade.py, run_autograder, setup.sh, requirements.txt   the harness
    local_autograder.py                                     imported by grade.py
    ref_double_pendulum.py, ref_plane.py, ref_answers.py    the answer key
    tests/inK.txt                                           case *inputs* only

No `outK.txt`: expected answers are computed from the reference at run time,
on perturbed inputs, so the fixtures' own outputs are of no use here.
"""

import argparse
import random
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE.parent
CODE = SOLUTIONS.parent / "code"  # the student handout

# The zip's root is what Gradescope unpacks into /autograder/source.
HARNESS = ["grade.py", "run_autograder", "setup.sh", "requirements.txt"]
FROM_CODE = ["local_autograder.py"]
FROM_SOLUTIONS = ["ref_double_pendulum.py", "ref_plane.py", "ref_answers.py"]

# (source fixture, name in the zip, rows to keep; None keeps all of them)
FIXTURES = [
    ("tests/in1.txt", "tests/in1.txt", None),
    ("tests/in2.txt", "tests/in2.txt", None),
    ("tests/in3.txt", "tests/in3.txt", None),
    ("tests/in4.txt", "tests/in4.txt", None),
]

EXECUTABLE = {"run_autograder", "setup.sh"}
_UNIX_EXEC = 0o755 << 16  # zip external_attr: rwxr-xr-x


def subsample(path, keep, rng):
    """`keep` lines from `path` by reservoir sampling, or all of them."""
    with open(path) as handle:
        lines = (line for line in handle if line.strip())
        if keep is None:
            return list(lines)
        reservoir = []
        for i, line in enumerate(lines):
            if len(reservoir) < keep:
                reservoir.append(line)
            else:
                j = rng.randrange(i + 1)
                if j < keep:
                    reservoir[j] = line
    return reservoir


def write_text(zf, name, text):
    """Add `text` at `name`, marking the shell entry points executable.

    Gradescope invokes run_autograder directly, so it has to come out of the
    zip with its exec bit; zipfile does not carry one over by default.
    """
    info = zipfile.ZipInfo(name)
    info.external_attr = _UNIX_EXEC if name in EXECUTABLE else (0o644 << 16)
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, text)


def build(output, seed, verbose=True):
    rng = random.Random(seed)
    missing = [
        p for p in (
            [HERE / n for n in HARNESS]
            + [CODE / n for n in FROM_CODE]
            + [SOLUTIONS / n for n in FROM_SOLUTIONS]
            + [CODE / src for src, _, _ in FIXTURES]
        )
        if not p.exists()
    ]
    if missing:
        raise SystemExit(
            "Missing:\n" + "\n".join(f"  {p}" for p in missing)
            + "\n\ntests/inK.txt come from:"
            "\n  python -m solutions.gen_ref_solution_data code"
        )

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in HARNESS:
            write_text(zf, name, (HERE / name).read_text())
        for name in FROM_CODE:
            write_text(zf, name, (CODE / name).read_text())
        for name in FROM_SOLUTIONS:
            write_text(zf, name, (SOLUTIONS / name).read_text())
        for src, dest, keep in FIXTURES:
            rows = subsample(CODE / src, keep, rng)
            write_text(zf, dest, "".join(rows))
            if verbose:
                print(f"  {dest}: {len(rows)} cases")

    size = Path(output).stat().st_size
    if verbose:
        print(f"\nwrote {output} ({size / 1e6:.2f} MB)")
        print("Upload it under Gradescope > Configure Autograder.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=str(HERE / "autograder.zip"))
    parser.add_argument("--seed", type=int, default=384,
                        help="subsampling seed, so the zip is reproducible "
                             "(default: %(default)s)")
    args = parser.parse_args()
    build(args.output, args.seed)


if __name__ == "__main__":
    main()
