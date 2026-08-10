"""The generated matrix has to stay deterministic, curated-first, and non-circular."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "tools/generate_validation_matrix.py"
RUNNER_PATH = ROOT / "tools/gpu_validation.py"
PROFILES_PATH = ROOT / "tests/gpu_validation/profiles.json"
CURATED_PATH = ROOT / "tests/gpu_validation/curated_cases.json"
MATRIX_PATH = ROOT / "tests/gpu_validation/matrix.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load("generate_validation_matrix", GENERATOR_PATH)


@pytest.fixture(scope="module")
def runner():
    return _load("gpu_validation_runner", RUNNER_PATH)


@pytest.fixture(scope="module")
def built(generator):
    return generator.build_matrix(PROFILES_PATH, CURATED_PATH)


@pytest.fixture(scope="module")
def matrix():
    return json.loads(MATRIX_PATH.read_text())


def test_the_checked_in_matrix_matches_the_declarations(generator, built, matrix):
    """A stale matrix would quietly plan against runner support that no longer exists."""
    generated, _ = built
    assert generated == matrix, (
        "tests/gpu_validation/matrix.json is stale; regenerate with "
        "python tools/generate_validation_matrix.py"
    )


def test_generation_is_deterministic(generator, built):
    again, _ = generator.build_matrix(PROFILES_PATH, CURATED_PATH)
    assert again == built[0]


def test_generation_does_not_depend_on_the_local_cache(
    generator, built, tmp_path, monkeypatch
):
    """Cases are a plan. Filtering them by local weights would make the file per-machine."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with_empty_cache, report = generator.build_matrix(PROFILES_PATH, CURATED_PATH)
    assert with_empty_cache == built[0]
    # The cache still decides what is reported as runnable here, just not what is planned.
    assert report["models_without_local_weights"] == report["planned_models"]


def test_generated_cases_never_assert_a_rejection(matrix):
    """Deriving a rejection from the code under test would assert only self-agreement."""
    for case in matrix["cases"]:
        if case["id"].startswith("gen-"):
            assert case["expected"]["outcome"] == "inference_success", case["id"]


def test_generated_cases_run_the_profile_runtime_args(matrix):
    """A sweep that leaves these unset measures a configuration nobody serves.

    Every eight-GPU Instinct entry in .ci/benchmark_configs pins an attention backend and enables
    torch.compile, so a generated case that omits them is not testing the deployed path.
    """
    profiles = json.loads(PROFILES_PATH.read_text())
    expected = {
        profile["id"]: profile.get("runtime_args", [])
        for profile in profiles["profiles"]
    }
    generated = [case for case in matrix["cases"] if case["id"].startswith("gen-")]
    assert generated
    for case in generated:
        profile_id = case["id"].split("-")[1]
        assert case["args"] == expected[profile_id], case["id"]


def test_a_model_needing_an_input_image_is_generated_once_its_sampling_names_one(
    generator, matrix
):
    """Filling in the image is the whole of the work; the cases come from the declarations.

    Before this, an image-to-image or image-to-video model was skipped outright, which left the
    edit and animate paths with no coverage at all on a node holding their weights.
    """
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
    from xfuser.model_executor.models.runner_models.loading.contracts import MaterializationMode

    profiles = json.loads(PROFILES_PATH.read_text())
    curated = json.loads(CURATED_PATH.read_text())
    needs_input = set(profiles["requires_input_image"]["models"])
    supplied = {
        model
        for model, settings in curated["sampling"].items()
        if settings.get("input_images")
        # A withheld model is generated nothing whatever its entry names, since every generated
        # case expects the load to succeed. Its entry exists for the curated case that asserts the
        # refusal, which needs the image only because the runner asks for one.
        and list(MODEL_REGISTRY[model].load_declaration.materialization_modes)
        != [MaterializationMode.EAGER]
    }
    assert supplied & needs_input, "no image-input model has an image, so this proves nothing"

    generated_models = {
        case["model"] for case in matrix["cases"] if case["id"].startswith("gen-")
    }
    for model in supplied & needs_input:
        assert model in generated_models, f"{model} has an input image but no generated case"


def test_a_model_needing_an_input_image_with_none_supplied_is_still_skipped(generator):
    """A case that passes no image would fail for reasons that say nothing about loading."""
    profiles = json.loads(PROFILES_PATH.read_text())
    registry = generator.load_registry()

    _, skipped = generator.generate_cases(registry, profiles, {})

    reasons = {entry["model"]: entry["reason"] for entry in skipped}
    withheld = [
        model
        for model in profiles["requires_input_image"]["models"]
        if reasons.get(model, "").startswith("needs an input image")
    ]
    assert withheld, "every image-input model was generated without an image being supplied"


def test_each_control_case_matches_the_rank_count_it_controls_for(matrix):
    """A control at a different rank count would not isolate anything.

    Spreading a model over more ranks moves host and device memory by itself, so the pair has to
    agree on world size for their difference to be attributable to the fill strategy.
    """
    controls = [
        case for case in matrix["cases"] if case["placement"] == "fsdp_eager_fill"
    ]
    assert controls, "the control placement stopped being generated"
    for control in controls:
        # Matched on rank count and offload rather than searched for and then
        # checked, because a curated blockwise case for the same model at another
        # rank count is not the pair this control belongs to
        counterpart = next(
            (
                case
                for case in matrix["cases"]
                if case["placement"] == "fsdp_blockwise"
                and case["model"] == control["model"]
                and case["quantization"] == control["quantization"]
                and case["world_size"] == control["world_size"]
                and case["offload"] == control["offload"]
            ),
            None,
        )
        assert counterpart is not None, control["id"]


def test_every_sharded_model_has_an_unsharded_case_at_the_same_rank_count(matrix):
    """The sharded cases have to be measured against what you would run instead of them.

    Eager at one rank is the image to compare against, but it is not a cost baseline: comparing an
    eight-rank sharded load to a one-rank load credits the loader for the parallelism. The honest
    comparison is eager at the same rank count, holding a full copy per rank.
    """
    # Curated cases are exempt: a hand-written case pins one configuration on one machine, and it
    # carries the note saying what it is for rather than a baseline beside it.
    sharded = [
        case
        for case in matrix["cases"]
        if case["placement"] == "fsdp_blockwise" and case["id"].startswith("gen-")
    ]
    assert sharded, "the blockwise placement stopped being generated"
    for case in sharded:
        if case["world_size"] == 1:
            continue
        assert any(
            baseline["placement"] == "eager"
            and baseline["model"] == case["model"]
            and baseline["world_size"] == case["world_size"]
            and baseline["quantization"] == "none"
            for baseline in matrix["cases"]
        ), f"{case['id']} has nothing unsharded to be compared against at its own rank count"


def test_a_baseline_at_the_served_rank_count_does_not_displace_the_reference(
    runner, matrix
):
    """Adding eager at eight ranks must not change which case others are scored against."""
    baselines = [
        case
        for case in matrix["cases"]
        if case["placement"] == "eager"
        and case["quantization"] == "none"
        and case["world_size"] > 1
    ]
    assert baselines, "the same-rank-count baseline stopped being generated"
    for baseline in baselines:
        reference = runner.reference_case_id(baseline, matrix["cases"])
        assert reference is not None, baseline["id"]
        assert next(
            case for case in matrix["cases"] if case["id"] == reference
        )["world_size"] == 1


def test_every_rejection_case_is_curated(matrix):
    curated = {case["id"] for case in json.loads(CURATED_PATH.read_text())["cases"]}
    for case in matrix["cases"]:
        if case["expected"]["outcome"] != "inference_success":
            assert case["id"] in curated, case["id"]


def test_curated_cases_survive_generation_unchanged(matrix):
    """Their IDs carry the result history, so regeneration must not rewrite them."""
    curated = json.loads(CURATED_PATH.read_text())["cases"]
    by_id = {case["id"]: case for case in matrix["cases"]}
    for case in curated:
        assert by_id[case["id"]] == case


def test_a_curated_case_supersedes_an_overlapping_generated_one(generator):
    """Two tokens admitting a common arch would both run there, so the pair is redundant."""
    curated = [
        {
            "id": "curated-one",
            "model": "M",
            "placement": "eager",
            "quantization": "fp8",
            "world_size": 1,
            "te_fp8": False,
            "offload": "none",
            "transformers": "5.x",
            "hardware": {"backend": "rocm_torchao", "accelerator": "non_rdna4_rocm"},
        }
    ]
    generated = [
        dict(curated[0], id="gen-overlapping"),
        dict(curated[0], id="gen-elsewhere"),
    ]
    generated[0]["hardware"] = {
        "backend": "rocm_torchao",
        "accelerator": "gfx942_or_gfx950",
    }
    generated[1]["hardware"] = {
        "backend": "rdna4_aiter",
        "accelerator": "gfx1200_or_gfx1201",
    }

    kept, superseded = generator.merge_cases(curated, generated)

    assert [case["id"] for case in superseded] == ["gen-overlapping"]
    assert [case["id"] for case in kept] == ["gen-elsewhere"]


@pytest.mark.parametrize(
    "left, right, overlaps",
    [
        ("gfx942_or_gfx950", "gfx950", True),
        ("gfx942_or_gfx950", "non_rdna4_rocm", True),
        ("gfx942", "gfx950", False),
        ("gfx942_or_gfx950", "gfx1200_or_gfx1201", False),
        ("gfx942_or_gfx950", "sm90", False),
    ],
)
def test_accelerator_overlap(generator, left, right, overlaps):
    assert generator.accelerators_overlap(left, right) is overlaps


def test_one_case_per_runner_not_per_registry_alias(generator, matrix):
    """Checkpoint ids are registered beside display names and must not each get a case."""
    registry = generator.load_registry()
    canonical = {alias for alias, _ in generator.canonical_runners(registry)}
    assert len(canonical) < len(registry)
    for case in matrix["cases"]:
        if case["id"].startswith("gen-"):
            assert case["model"] in canonical, case["model"]


def test_the_generated_matrix_passes_the_runner_validation(runner, matrix):
    runner.validate_matrix(matrix)


def test_generated_ids_are_unique_and_well_formed(runner, matrix):
    ids = [case["id"] for case in matrix["cases"]]
    assert len(ids) == len(set(ids))
    for case_id in ids:
        assert runner.ID_PATTERN.fullmatch(case_id), case_id
