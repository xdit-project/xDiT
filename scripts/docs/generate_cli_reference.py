"""Render the runner's registered argparse options as checked-in Markdown.

Run in an installed xDiT environment. Importing this module and testing the
renderer only require the Python standard library; xDiT is imported in main.
"""

import argparse
import difflib
import html
import re
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "runner" / "cli.md"

GROUP_ORDER = (
    "Model selection and adapters",
    "Generation inputs",
    "Parallelism and PipeFusion",
    "Loading and CPU offload",
    "VAE memory and layout",
    "Compilation and GEMM quantization",
    "Attention backends and communication",
    "Sparse attention",
    "Step caching",
    "Benchmarking, profiling, and output",
    "Other options",
)


def _category(action: argparse.Action) -> str:
    """Presentation only: unknown options remain visible without schema updates."""
    name = action.option_strings[0].lstrip("-").replace("-", "_")
    if name in {"model", "task"} or name.startswith(("lora_", "vdn_", "distilled_transformer")):
        return "Model selection and adapters"
    if name in {
        "height",
        "width",
        "num_frames",
        "prompt",
        "negative_prompt",
        "num_inference_steps",
        "max_sequence_length",
        "seed",
        "guidance_scale",
        "guidance_scale_2",
        "flow_shift",
        "input_images",
        "resize_input_images",
        "audio_prompt",
        "batch_size",
        "dataset_path",
    }:
        return "Generation inputs"
    if name.endswith("_degree") or name in {
        "use_cfg_parallel",
        "num_pipeline_patch",
        "attn_layer_num_for_pp",
        "warmup_steps",
        "fully_shard_components",
        "no_reshard_after_forward",
    }:
        return "Parallelism and PipeFusion"
    if name.startswith(("memory_efficient_", "group_offload_")) or name in {
        "enable_sequential_cpu_offload",
        "enable_model_cpu_offload",
        "enable_group_cpu_offload",
    }:
        return "Loading and CPU offload"
    if name.startswith(("vae_", "use_vae_")) or name in {"use_parallel_vae", "enable_tiling", "enable_slicing"}:
        return "VAE memory and layout"
    if "gemm" in name or name in {"use_torch_compile", "use_fp8_text_encoder"} or name.startswith("fp8_precision_"):
        return "Compilation and GEMM quantization"
    if name.startswith(("attention_", "cross_attention_", "mha_", "hybrid_attn_", "fp8_comms_")) or name in {
        "use_hybrid_attn_schedule",
        "num_hybrid_attn_high_precision_steps",
        "use_fp8_comms",
    }:
        return "Attention backends and communication"
    if name.startswith(("spargeattn_", "use_spargeattn_", "sol_attn_", "vsa_", "use_vsa_", "bsa_", "use_ssta_")):
        return "Sparse attention"
    if name.startswith("cache_") or name in {"use_teacache", "use_fbcache"}:
        return "Step caching"
    if name.startswith(("profile", "determinism_check")) or name in {
        "num_iterations",
        "warmup_calls",
        "output_directory",
    }:
        return "Benchmarking, profiling, and output"
    return "Other options"


def _code(text: str) -> str:
    """Quote a value without letting embedded backticks end the code span."""
    runs = re.findall(r"`+", text)
    fence = "`" * (max(map(len, runs), default=0) + 1)
    text = text.replace("|", r"\|")
    return f"{fence} {text} {fence}" if runs or text.startswith(" ") or text.endswith(" ") else f"{fence}{text}{fence}"


def _prose(text: str) -> str:
    text = html.escape(" ".join(text.split()), quote=False)
    return re.sub(r"([\\`*_\[\]#|])", r"\\\1", text)


def _long_choices(action: argparse.Action) -> bool:
    return action.choices is not None and len(",".join(str(choice) for choice in action.choices)) > 48


def _syntax(action: argparse.Action) -> str:
    options = "<br>".join(_code(option) for option in action.option_strings)
    if action.nargs == 0:
        return options
    value = action.metavar or {str: "TEXT", int: "INT", float: "FLOAT"}.get(action.type or str, "VALUE")
    if isinstance(value, tuple):
        return options + " " + _code(" ".join(value))
    if action.choices is not None:
        value = "CHOICE" if _long_choices(action) else "{" + ",".join(str(choice) for choice in action.choices) + "}"
    forms = {None: value, "?": f"[{value}]", "*": f"[{value} ...]", "+": f"{value} [{value} ...]"}
    form = forms.get(action.nargs)
    if form is None:
        form = " ".join([value] * action.nargs) if isinstance(action.nargs, int) else f"{value} ..."
    return options + " " + _code(form)


def render_reference(parser: argparse.ArgumentParser) -> str:
    """Render registered options without parsing arguments or running xDiT."""
    groups = {title: [] for title in GROUP_ORDER}
    for action in parser._actions:
        if action.option_strings and not isinstance(action, argparse._HelpAction) and action.help != argparse.SUPPRESS:
            groups[_category(action)].append(action)
    lines = [
        "# Runner CLI reference",
        "",
        "<!-- Generated by scripts/docs/generate_cli_reference.py; edit parser help in xfuser/config/args.py. -->",
        "",
        "This page is generated from the runner's argument parser. See the",
        "[runner guide](runner.md) for launch options, examples, model defaults,",
        "and compatibility restrictions. Dedicated model examples use a different parser.",
        "",
        "**Parser defaults are not necessarily effective model defaults.** An omitted",
        "value may be resolved by the selected model or configuration validation;",
        "`None` does not by itself mean that a feature is disabled. A registered",
        "option is not a promise that every model or device supports it.",
        "",
        "Runner long options accept underscores or dashes (for example,",
        "`--ulysses_degree` and `--ulysses-degree`). The spellings below are those",
        "registered by argparse, including negative spellings for Boolean options.",
        "",
        "In the syntax column, `TEXT`, `INT`, and `FLOAT` name value types; `VALUE`",
        "uses the format described by the option's help.",
        "`CHOICE` refers to the accepted values listed in the description.",
        "Brackets mean optional values:",
        "`[TEXT ...]` accepts zero or more, and `TEXT [TEXT ...]` accepts one or more.",
        "Flags without a value placeholder take no value. Boolean positive spellings",
        "enable a setting and their `--no-...` spellings disable it.",
        "",
    ]
    lines.extend(
        f"- [{title}](#{title.lower().replace(',', '').replace(' ', '-')})"
        for title, actions in groups.items()
        if actions
    )
    lines.append("")
    formatter = parser._get_formatter()
    for title, actions in groups.items():
        if not actions:
            continue
        lines.extend([f"## {title}", "", "| Syntax | Parser default | Description |", "| --- | --- | --- |"])
        for action in actions:
            # Match Python-Markdown's former heading IDs, so existing flag links
            # continue to work after replacing per-option headings with rows.
            anchor = re.sub(r"[-\s]+", "-", action.option_strings[0].lower())
            anchors = dict.fromkeys((anchor, action.option_strings[0].lower()))
            syntax = "".join(f'<a id="{html.escape(value, quote=True)}"></a>' for value in anchors) + _syntax(action)
            default = "Absent" if action.default == argparse.SUPPRESS else _code(repr(action.default))
            if action.dest != action.option_strings[0].lstrip("-").replace("-", "_"):
                default += "<br>Setting: " + _code(action.dest)
            description = _prose(formatter._expand_help(action)) if action.help else "No additional parser help."
            if action.required:
                description = "**Required.** " + description
            if isinstance(action, argparse._StoreFalseAction):
                description += " Sets " + _code(action.dest) + " to `False`."
            if _long_choices(action):
                description += "<br>Choices: " + ", ".join(_code(str(choice)) for choice in action.choices) + "."
            lines.append(f"| {syntax} | {default} | {description} |")
        lines.append("")
    return "\n".join(lines)


def update_reference(parser: argparse.ArgumentParser, output: Path, *, check: bool = False) -> int:
    """Write a reference, or report drift without changing an existing file."""
    generated = render_reference(parser)
    if not check:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(generated, encoding="utf-8")
        print(f"Wrote {output}")
        return 0
    existing = output.read_text(encoding="utf-8") if output.exists() else ""
    if generated == existing:
        print(f"CLI reference is current: {output}")
        return 0
    print(
        "".join(
            difflib.unified_diff(
                existing.splitlines(keepends=True),
                generated.splitlines(keepends=True),
                fromfile=str(output),
                tofile="generated reference",
            )
        ),
        end="",
    )
    print("Regenerate in your installed xDiT environment: python -m scripts.docs.generate_cli_reference")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the checked-in reference is stale; do not write files."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Reference Markdown path.")
    args = parser.parse_args(argv)

    # Registration imports the installed package, but does not initialize the
    # distributed environment, instantiate a model, or invoke the runner CLI.
    from xfuser.config.args import FlexibleArgumentParser, xFuserArgs

    runner_parser = xFuserArgs.add_runner_args(FlexibleArgumentParser(prog="xdit", description="xFuser Arguments"))
    return update_reference(runner_parser, args.output, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
