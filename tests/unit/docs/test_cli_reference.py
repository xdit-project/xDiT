"""Exercise generated documentation using real argparse actions, without xDiT imports."""

import argparse

from scripts.docs.generate_cli_reference import render_reference, update_reference


def test_options_preserve_value_forms_and_escape_help():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["small", "large"], help="Select <model> & preserve [text|value]."
    )
    parser.add_argument("--images", nargs="+", default=[], help="One or more input images.")
    parser.add_argument("--prompt", nargs="*", help="Prompt with `code` and_underlines.")
    parser.add_argument("--sparse", action=argparse.BooleanOptionalAction, default=True, help="Use sparse attention.")
    parser.add_argument("--keep", action="store_false", dest="reshard", help="Keep gathered weights.")

    reference = render_reference(parser)

    assert reference.index("## Model selection and adapters") < reference.index("## Generation inputs")
    assert reference.index('id="-model"') < reference.index('id="-images"') < reference.index('id="-sparse"')
    assert 'id="--model"' in reference
    assert "`--model` `{small,large}` | `None` | **Required.**" in reference
    assert "`--images` `TEXT [TEXT ...]` | `[]`" in reference
    assert "`--prompt` `[TEXT ...]`" in reference
    assert "`--sparse`<br>`--no-sparse` | `True`" in reference
    assert "Setting: `reshard`" in reference
    assert "Sets `reshard` to `False`." in reference
    assert "Select &lt;model&gt; &amp; preserve \\[text\\|value\\]." in reference
    assert "Prompt with \\`code\\` and\\_underlines." in reference
    assert 'id="-help"' not in reference


def test_syntax_types_optional_values_and_unknown_options_remain_visible():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, help="Output height.")
    parser.add_argument("--flow_shift", type=float, help="Scheduler shift.")
    parser.add_argument("--future_option", nargs="?", choices=["a|b", "c"], default="a|b", help="Future | option.")
    parser.add_argument("--coordinates", nargs=2, type=float, metavar=("X", "Y"), help="Coordinates.")

    reference = render_reference(parser)

    assert "`--height` `INT`" in reference
    assert "`--flow_shift` `FLOAT`" in reference
    assert "## Other options" in reference
    assert "`--future_option` `[{a\\|b,c}]` | `'a\\|b'` | Future \\| option." in reference
    assert "`--coordinates` `X Y`" in reference


def test_long_choices_wrap_individually_without_widening_syntax():
    parser = argparse.ArgumentParser()
    choices = ["none", "auto", "e4m3-e4m3", "int8-e4m3", "mxfp8-e4m3", "e4m3-mxfp6", "mxfp6-e4m3"]
    parser.add_argument("--attention_a2a", choices=choices, default="none", help="Input transport profile.")

    reference = render_reference(parser)
    row = next(line for line in reference.splitlines() if 'id="--attention_a2a"' in line)
    syntax, default, description = row.strip("|").split(" | ")

    assert "`--attention_a2a` `CHOICE`" in syntax
    assert default == "`'none'`"
    assert "Input transport profile." in description
    assert "Choices: " + ", ".join(f"`{choice}`" for choice in choices) in description
    assert "{" not in syntax


def test_boolean_help_is_stable_across_argparse_versions():
    references = []
    for help_text in ("Use sparse attention.", "Use sparse attention. (default: True)"):
        parser = argparse.ArgumentParser()
        action = parser.add_argument("--sparse", action=argparse.BooleanOptionalAction, default=True)
        action.help = help_text  # Stored help differs between Python 3.10 and newer Python.
        references.append(render_reference(parser))

    assert references[0] == references[1]
    assert "`--sparse`<br>`--no-sparse` | `True` | Use sparse attention." in references[0]


def test_check_reports_drift_without_writing(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="a`b", help="Destination (default: %(default)s).")
    output = tmp_path / "docs" / "cli.md"

    assert update_reference(parser, output, check=True) == 1
    assert not output.exists()
    capsys.readouterr()
    assert update_reference(parser, output) == 0
    generated = output.read_bytes()
    assert b"`` 'a`b' ``" in generated
    assert update_reference(parser, output, check=True) == 0
    assert output.read_bytes() == generated

    parser.add_argument("--count", type=int, default=2, help="How many outputs to save.")
    assert update_reference(parser, output, check=True) == 1
    assert output.read_bytes() == generated
    assert 'id="-count"' in capsys.readouterr().out
    assert update_reference(parser, output) == 0
    assert update_reference(parser, output, check=True) == 0


def test_suppressed_options_and_defaults(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible", default=argparse.SUPPRESS, help="Only present when supplied.")
    parser.add_argument("--hidden", help=argparse.SUPPRESS)
    output = tmp_path / "cli.md"

    update_reference(parser, output)
    reference = output.read_text()

    assert "`--visible` `TEXT` | Absent" in reference
    assert "--hidden" not in reference
