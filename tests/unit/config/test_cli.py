from types import SimpleNamespace

import pytest

from xfuser import cli
from xfuser.cli import get_nproc_from_args


def test_text_encoder_tp_reuses_model_ranks():
    args = [
        "--ulysses_degree",
        "8",
        "--text_encoder_tp_degree",
        "8",
    ]

    assert get_nproc_from_args(args) == 8


@pytest.mark.parametrize("cfg_flag", ["--use_cfg_parallel", "--use-cfg-parallel"])
def test_cfg_parallel_doubles_the_processes_in_either_spelling(cfg_flag):
    assert get_nproc_from_args(["--model", "m", "--ulysses-degree", "2", cfg_flag]) == 4


def test_degrees_multiply_across_spellings_and_value_forms():
    args = ["--ulysses_degree=2", "--ring-degree", "2", "--data-parallel-degree=2", "--prompt", "a cat"]

    assert get_nproc_from_args(args) == 8


def _launch(monkeypatch, args):
    """Run the CLI with torchrun replaced by a recorder; return its command."""
    launched = []

    def fake_popen(cmd, **kwargs):
        launched.append(cmd)
        return SimpleNamespace(pid=0, wait=lambda: 0)

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args)
    return launched, exit_info.value


def test_multi_node_launch_splits_the_processes_across_nodes(monkeypatch):
    launched, exit_status = _launch(
        monkeypatch,
        ["--nnodes", "2", "--node_rank", "1", "--model", "m", "--ulysses_degree", "8"],
    )

    assert exit_status.code == 0
    (cmd,) = launched
    assert "--nproc_per_node=4" in cmd
    assert "--nnodes=2" in cmd


def test_multi_node_launch_rejects_degrees_the_nodes_cannot_share(monkeypatch):
    launched, exit_status = _launch(
        monkeypatch,
        ["--nnodes=3", "--model", "m", "--ulysses_degree", "8"],
    )

    assert launched == []
    assert "--nnodes=3" in str(exit_status.code)


def test_explicit_nproc_per_node_is_kept(monkeypatch):
    launched, _ = _launch(
        monkeypatch,
        ["--nnodes", "2", "--nproc_per_node", "8", "--model", "m", "--ulysses_degree", "8"],
    )

    assert "--nproc_per_node=8" in launched[0]


def test_torchrun_flags_are_accepted_with_dashes(monkeypatch):
    launched, _ = _launch(
        monkeypatch,
        ["--nnodes", "2", "--node-rank", "1", "--master-addr=10.0.0.1", "--model", "m", "--ulysses-degree", "8"],
    )

    (cmd,) = launched
    assert "--node_rank=1" in cmd
    assert "--master_addr=10.0.0.1" in cmd
    assert cmd[cmd.index("xfuser.runner") + 1 :] == ["--model", "m", "--ulysses-degree", "8"]


def test_a_malformed_degree_is_reported_as_an_xdit_usage_error(monkeypatch, capsys):
    launched, exit_status = _launch(monkeypatch, ["--model", "m", "--ulysses_degree", "two"])

    assert launched == []
    assert exit_status.code == 2
    assert "xdit: error: argument --ulysses_degree" in capsys.readouterr().err
