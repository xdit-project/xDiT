from xfuser.cli import get_nproc_from_args


def test_text_encoder_tp_reuses_model_ranks():
    args = [
        "--ulysses_degree",
        "8",
        "--text_encoder_tp_degree",
        "8",
    ]

    assert get_nproc_from_args(args) == 8
