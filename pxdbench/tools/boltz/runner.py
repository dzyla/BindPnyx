"""Subprocess driver for Boltz-2.

Boltz pins a different torch than the pipeline does, so it gets its own
environment and is invoked as a subprocess, the way `vanilla_mpnn_predictor`
already shells out. `pxdbench.toolenv` finds it; nothing here assumes a path.

One invocation per SEED over a directory of YAMLs: `check_inputs`
(boltz/main.py:281) expands a directory with `glob("*")` and `process_inputs`
(main.py:795) processes every path it returns, so batching the directory turns N
model loads into one.

`out_dir` must never be the YAML directory: `check_inputs` (main.py:300-311)
raises on any subdirectory inside the input directory, so a later seed would die
on an earlier seed's `boltz_results_*`.
"""
import os
import subprocess

from pxdbench.toolenv import resolve_boltz_bin  # noqa: F401  (re-exported)


# msa_check is stdlib-only and lives in this same package, so importing the
# constant adds no package initialisation; one definition cannot drift.
from pxdbench.tools.boltz.msa_check import DUMMY_MSA_WARNING as _DUMMY_MSA_WARNING  # noqa: E402

#: Boltz reports per-example failures with print() on STDOUT and exits 0.
FAILED_EXAMPLES_MARKER = "Number of failed examples"


def _text(stream):
    """TimeoutExpired carries bytes or str depending on the call."""
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode(errors="replace")


def build_command(
    boltz_bin,
    yaml_dir,
    out_dir,
    seed,
    recycling_steps=3,
    diffusion_samples=1,
    num_workers=0,
    no_kernels=True,
):
    """The Boltz-2 command line for one seed over a whole YAML directory.

    `num_workers` is Boltz's dataloader worker count. 0 is the default and is
    what run_boltz_campaign.py used, so it is also what the calibration was
    produced under. Raising it overlaps featurisation with compute inside a
    single invocation, which is where much of the GPU idle time goes - but it
    must be shown not to change any score before being used, because a Boltz
    score is sensitive to how its invocation is composed.
    """
    cmd = [
        boltz_bin,
        "predict",
        yaml_dir,
        "--out_dir",
        out_dir,
        "--diffusion_samples",
        str(diffusion_samples),
        "--recycling_steps",
        str(recycling_steps),
        "--output_format",
        "pdb",
        "--write_full_pae",
        "--num_workers",
        str(int(num_workers)),
        "--seed",
        str(seed),
        "--override",
    ]
    if no_kernels:
        # --no_kernels disables Boltz's optimised triangular attention, which
        # routes through cuequivariance_torch. run_boltz_campaign.py passed it,
        # so the calibration was produced with the kernels OFF. Turning them on
        # is a speed lever that must be shown not to change scores first.
        cmd.insert(cmd.index("--seed"), "--no_kernels")
    return cmd


def run_seed(
    boltz_bin,
    yaml_dir,
    out_dir,
    seed,
    timeout_s=3600,
    gpu="0",
    recycling_steps=3,
    diffusion_samples=1,
    num_workers=0,
    no_kernels=True,
):
    """Run one seed. Never raises on a Boltz failure; reports it."""
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = build_command(
        boltz_bin,
        yaml_dir,
        out_dir,
        seed,
        recycling_steps=recycling_steps,
        diffusion_samples=diffusion_samples,
        num_workers=num_workers,
        no_kernels=no_kernels,
    )
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        returncode = proc.returncode
        stdout, stderr = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        # Keep whatever was printed before the hang: it names the stage.
        returncode = 124
        stdout = _text(exc.stdout)
        stderr = f"timeout after {timeout_s}s\n" + _text(exc.stderr)

    # Both streams are kept separately. Boltz reports per-example failures and
    # a discarded MSA on STDOUT and still exits 0, so collapsing the two - the
    # old `proc.stderr or proc.stdout` - threw away the only report of either
    # whenever anything at all had been written to stderr.
    combined = (stdout + stderr) if (stdout or stderr) else ""
    # Scan the FULL stream, not a tail. Boltz prints the dummy-MSA line once,
    # early, in the middle of a progress bar; thousands of later progress
    # characters push it out of any fixed-size tail. Counting here is the only
    # place the whole output exists.
    dummy_hits = combined.count(_DUMMY_MSA_WARNING)
    dummy_lines = [
        line for line in combined.splitlines()
        if _DUMMY_MSA_WARNING in line
    ][:8]
    # Same reasoning as above: extract from the full stream. stdout comes first
    # in `combined`, so a long stderr (torch noise) pushes this line out of any
    # tail window.
    failed_lines = [
        line.strip() for line in combined.splitlines()
        if FAILED_EXAMPLES_MARKER in line
    ][:8]
    return {
        "returncode": returncode,
        "stderr_tail": (stderr or stdout)[-2000:],
        "stderr_raw_tail": stderr[-2000:],
        "failed_example_lines": failed_lines,
        "stdout_tail": stdout[-4000:],
        "combined_tail": combined[-8000:],
        "msa_dummy_count": dummy_hits,
        "msa_dummy_lines": dummy_lines,
        "out_dir": out_dir,
    }
