import os
import stat

from pxdbench.tools.boltz.runner import build_command, run_seed


def test_command_is_batched_over_a_directory(tmp_path):
    """The YAML DIRECTORY is the argument, so the model loads once per seed."""
    cmd = build_command("/bin/boltz", str(tmp_path), str(tmp_path / "out"), seed=3)
    assert cmd[:2] == ["/bin/boltz", "predict"]
    assert cmd[2] == str(tmp_path)


def test_command_carries_the_calibrated_flags(tmp_path):
    cmd = build_command("/bin/boltz", str(tmp_path), str(tmp_path / "out"), seed=3)
    joined = " ".join(cmd)
    assert "--write_full_pae" in joined
    assert "--output_format pdb" in joined
    assert "--diffusion_samples 1" in joined
    assert "--recycling_steps 3" in joined
    assert "--seed 3" in joined
    assert "--override" in joined
    assert "--no_kernels" in joined
    assert "--use_msa_server" not in joined, "MSAs are cached, never fetched"


def test_out_dir_is_never_the_yaml_dir(tmp_path):
    """check_inputs raises on a subdirectory inside the input dir, so a later
    seed would die on an earlier seed's boltz_results_* if these collided."""
    yaml_dir = str(tmp_path / "input")
    out_dir = str(tmp_path)
    cmd = build_command("/bin/boltz", yaml_dir, out_dir, seed=1)
    assert cmd[2] == yaml_dir
    assert cmd[cmd.index("--out_dir") + 1] == out_dir
    assert cmd[cmd.index("--out_dir") + 1] != cmd[2]


def test_no_foreign_environment_path_is_baked_in():
    """The regression that made the app unusable on any other machine.

    runner.py shipped DEFAULT_BOLTZ_BIN pointing at one person's unrelated conda
    env, so a fresh machine ran diffusion and MPNN to completion and then died
    at the first fold with FileNotFoundError.
    """
    import inspect

    from pxdbench.tools.boltz import runner

    src = inspect.getsource(runner)
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "boltz2local" not in code
    assert "/home/" not in code


def test_run_seed_reports_failure_without_raising(tmp_path):
    stub = tmp_path / "fake_boltz"
    stub.write_text("#!/bin/sh\necho 'boom' >&2\nexit 9\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    result = run_seed(str(stub), str(tmp_path), str(tmp_path / "out"), seed=1)
    assert result["returncode"] == 9
    assert "boom" in result["stderr_tail"]


def test_run_seed_reports_timeout_as_124(tmp_path):
    stub = tmp_path / "slow_boltz"
    stub.write_text("#!/bin/sh\nsleep 5\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    result = run_seed(
        str(stub), str(tmp_path), str(tmp_path / "out"), seed=1, timeout_s=1
    )
    assert result["returncode"] == 124
    assert "timeout" in result["stderr_tail"]


def test_run_seed_sets_cuda_visible_devices(tmp_path):
    stub = tmp_path / "fake_boltz"
    # argv is [boltz, predict, <yaml_dir>, ...] so the yaml dir is $2, not $1
    stub.write_text('#!/bin/sh\necho "$CUDA_VISIBLE_DEVICES" > "$2"/seen.txt\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    run_seed(str(stub), str(tmp_path), str(tmp_path / "out"), seed=1, gpu="2")
    assert (tmp_path / "seen.txt").read_text().strip() == "2"


def test_run_seed_passes_the_yaml_dir_as_the_positional_arg(tmp_path):
    """Pins argv order, which the CUDA test above depends on."""
    stub = tmp_path / "fake_boltz"
    stub.write_text('#!/bin/sh\necho "$1 $2" > "$2"/argv.txt\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    yaml_dir = tmp_path / "input"
    yaml_dir.mkdir()
    run_seed(str(stub), str(yaml_dir), str(tmp_path / "out"), seed=1)
    assert (yaml_dir / "argv.txt").read_text().strip() == f"predict {yaml_dir}"


def test_run_seed_creates_the_out_dir(tmp_path):
    stub = tmp_path / "fake_boltz"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "nested" / "out"
    run_seed(str(stub), str(tmp_path), str(out), seed=1)
    assert out.is_dir()


def test_run_seed_keeps_stdout_even_when_stderr_is_nonempty(tmp_path):
    """boltz reports per-example failures and discarded MSAs on STDOUT, and
    exits 0. Folding stdout into stderr lost exactly that."""
    stub = tmp_path / "b"
    stub.write_text(
        "#!/bin/sh\n"
        "echo 'Number of failed examples: 64'\n"
        "echo 'some torch warning' >&2\n"
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    result = run_seed(str(stub), str(tmp_path), str(tmp_path / "o"), seed=1)
    assert result["returncode"] == 0
    assert "Number of failed examples: 64" in result["stdout_tail"]
    assert "some torch warning" in result["stderr_tail"]
    assert "Number of failed examples: 64" in result["combined_tail"]
    assert "some torch warning" in result["combined_tail"]


def test_the_existing_keys_are_unchanged(tmp_path):
    """Four call sites read returncode and stderr_tail by subscript."""
    stub = tmp_path / "b"
    stub.write_text("#!/bin/sh\necho boom >&2\nexit 9\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    result = run_seed(str(stub), str(tmp_path), str(tmp_path / "o"), seed=1)
    assert result["returncode"] == 9
    assert "boom" in result["stderr_tail"]
    assert result["out_dir"] == str(tmp_path / "o")


def test_a_timeout_still_reports_124_on_both_streams(tmp_path):
    stub = tmp_path / "b"
    stub.write_text("#!/bin/sh\nsleep 30\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    result = run_seed(str(stub), str(tmp_path), str(tmp_path / "o"),
                      seed=1, timeout_s=1)
    assert result["returncode"] == 124
    assert "timeout" in result["stderr_tail"]
    assert "timeout" in result["combined_tail"]


def test_the_dummy_msa_warning_is_found_behind_a_long_progress_stream(tmp_path):
    """It is printed once, early, then buried. An 8KB tail would miss it, which
    is why run_seed counts on the full stream before truncating."""
    stub = tmp_path / "b"
    stub.write_text(
        "#!/bin/sh\n"
        "echo 'Warning: MSA does not match input sequence, creating dummy. 1'\n"
        "i=0; while [ $i -lt 400 ]; do "
        "echo '....................................................'; "
        "i=$((i+1)); done\n"
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    result = run_seed(str(stub), str(tmp_path), str(tmp_path / "o"), seed=1)
    assert result["msa_dummy_count"] == 1
    assert "creating dummy" in result["msa_dummy_lines"][0]
    # and prove the tail really would have missed it
    assert "creating dummy" not in result["combined_tail"]


def test_a_clean_run_counts_no_dummy_warnings(tmp_path):
    stub = tmp_path / "b"
    stub.write_text("#!/bin/sh\necho 'all good'\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    result = run_seed(str(stub), str(tmp_path), str(tmp_path / "o"), seed=1)
    assert result["msa_dummy_count"] == 0
    assert result["msa_dummy_lines"] == []


def test_one_warning_per_affected_chain_is_counted(tmp_path):
    stub = tmp_path / "b"
    stub.write_text(
        "#!/bin/sh\n"
        "echo 'Warning: MSA does not match input sequence, creating dummy. 1'\n"
        "echo 'Warning: MSA does not match input sequence, creating dummy. 2'\n"
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    assert run_seed(str(stub), str(tmp_path), str(tmp_path / "o"),
                    seed=1)["msa_dummy_count"] == 2


def test_partial_output_survives_a_timeout(tmp_path):
    """What boltz printed before it hung is the diagnostic."""
    stub = tmp_path / "b"
    stub.write_text("#!/bin/sh\necho 'processed 3 of 64'\nsleep 30\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    result = run_seed(str(stub), str(tmp_path), str(tmp_path / "o"),
                      seed=1, timeout_s=2)
    assert result["returncode"] == 124
    assert "processed 3 of 64" in result["combined_tail"]
