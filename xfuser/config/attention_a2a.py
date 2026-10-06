from __future__ import annotations

from dataclasses import dataclass

_PROFILE_RECIPES = {
    "e4m3-e4m3": {
        "codecs": ("e4m3", "e4m3", "e4m3"),
        "consumer_codecs": ("e4m3", "e4m3", "e4m3"),
        "scale_modes": ("f32_per_tensor",) * 3,
        "attention_backend": "aiter_fp8",
        "v_pack": "default",
        "pad_multiple": 32,
        "default_hadamard": "preprocess",
    },
    "int8-e4m3": {
        "codecs": ("int8", "int8", "e4m3"),
        "consumer_codecs": ("int8", "int8", "e4m3"),
        "scale_modes": ("f32_per_tensor",) * 3,
        "attention_backend": "aiter_i8fp8",
        "v_pack": "default",
        "pad_multiple": 32,
        "default_hadamard": "none",
    },
    "mxfp8-e4m3": {
        "codecs": ("mxfp8", "mxfp8", "e4m3"),
        "consumer_codecs": ("e4m3", "e4m3", "e4m3"),
        "scale_modes": (
            "e8m0_per_1x32",
            "e8m0_per_1x32",
            "f32_per_tensor",
        ),
        "attention_backend": "aiter_mxfp8",
        "v_pack": "default",
        "pad_multiple": 32,
        "default_hadamard": "preprocess",
    },
    "e4m3-mxfp6": {
        "codecs": ("e4m3", "e4m3", "mxfp6_p"),
        "consumer_codecs": ("e4m3", "e4m3", "mxfp6"),
        "scale_modes": (
            "f32_per_tensor",
            "f32_per_tensor",
            "e8m0_per_1x32",
        ),
        "attention_backend": "aiter_f8f6",
        "v_pack": "fp6_p",
        "pad_multiple": 64,
        "default_hadamard": "preprocess",
    },
    "mxfp6-e4m3": {
        "codecs": ("mxfp6", "mxfp6", "e4m3_pc"),
        "consumer_codecs": ("mxfp6", "mxfp6", "e4m3"),
        "scale_modes": (
            "e8m0_per_1x32",
            "e8m0_per_1x32",
            "f32_per_channel",
        ),
        "attention_backend": "aiter_mxfp6",
        "v_pack": "default",
        "pad_multiple": 32,
        "default_hadamard": "preprocess",
    },
    "mxfp6-mxfp6": {
        "codecs": ("mxfp6", "mxfp6", "mxfp6_p"),
        "consumer_codecs": ("mxfp6", "mxfp6", "mxfp6"),
        "scale_modes": ("e8m0_per_1x32",) * 3,
        "attention_backend": "aiter_f6f6",
        "v_pack": "fp6_p",
        "pad_multiple": 64,
        "default_hadamard": "preprocess",
    },
    "mxfp4-mxfp4": {
        "codecs": ("mxfp4", "mxfp4", "mxfp4"),
        "consumer_codecs": ("mxfp4", "mxfp4", "mxfp4"),
        "scale_modes": ("e8m0_per_1x32",) * 3,
        "attention_backend": "aiter_mxfp4",
        "v_pack": "fp6_p",
        "pad_multiple": 64,
        "default_hadamard": "preprocess",
    },
    "mxfp6-mxfp4": {
        "codecs": ("mxfp6", "mxfp6", "mxfp4"),
        "consumer_codecs": ("mxfp6", "mxfp6", "mxfp4"),
        "scale_modes": ("e8m0_per_1x32",) * 3,
        "attention_backend": "aiter_f6f4",
        "v_pack": "fp6_p",
        "pad_multiple": 64,
        "default_hadamard": "preprocess",
    },
}

_BACKEND_PROFILES = {
    recipe["attention_backend"]: profile for profile, recipe in _PROFILE_RECIPES.items() if recipe.get("auto", True)
}
_BACKEND_PROFILES["aiter_f4f4"] = "mxfp4-mxfp4"
_UNSUPPORTED_BACKEND_REASONS = {
    "aiter_mxfp4_sparge": "Attention A2A currently supports dense MHA-v4 only",
}

ATTENTION_A2A_PROFILES = ("none", "auto", *_PROFILE_RECIPES)
ATTENTION_A2A_HADAMARD_PLACEMENTS = (
    "auto",
    "preprocess",
    "transport",
    "epilogue",
    "none",
)


@dataclass(frozen=True)
class AttentionA2AConfig:
    """Validated configuration for the dense Ulysses Attention A2A input hop."""

    profile: str = "none"
    hadamard: str = "auto"

    def __post_init__(self) -> None:
        profile = self.profile.strip().lower()
        hadamard = self.hadamard.strip().lower()
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "hadamard", hadamard)

        if profile not in ATTENTION_A2A_PROFILES:
            raise ValueError(
                f"unsupported Attention A2A profile {profile!r}; expected one of {', '.join(ATTENTION_A2A_PROFILES)}"
            )
        if hadamard not in ATTENTION_A2A_HADAMARD_PLACEMENTS:
            raise ValueError(
                f"unsupported Attention A2A Hadamard placement {hadamard!r}; "
                f"expected one of {', '.join(ATTENTION_A2A_HADAMARD_PLACEMENTS)}"
            )
        if profile == "none" and hadamard != "auto":
            raise ValueError("--attention_a2a_hadamard requires an enabled --attention_a2a profile")
        if profile == "int8-e4m3" and hadamard == "transport":
            raise ValueError(
                "INT8 Attention A2A cannot apply Hadamard in transport; use "
                "preprocess to rotate explicitly, or auto/none for the native recipe"
            )

    @property
    def enabled(self) -> bool:
        return self.profile != "none"

    @property
    def is_auto(self) -> bool:
        return self.profile == "auto"

    def resolve_for_backend(self, backend) -> AttentionA2AConfig:
        """Resolve ``auto`` and validate explicit profiles against one backend."""
        if not self.enabled:
            return self
        backend_name = backend.name.lower() if hasattr(backend, "name") else str(backend).strip().lower()
        if self.is_auto:
            try:
                profile = _BACKEND_PROFILES[backend_name]
            except KeyError:
                detail = _UNSUPPORTED_BACKEND_REASONS.get(backend_name)
                raise ValueError(
                    f"attention backend {backend_name!r} has no packed Attention "
                    "A2A recipe" + (f": {detail}" if detail else "")
                ) from None
            return AttentionA2AConfig(profile=profile, hadamard=self.hadamard)
        compatible_backends = {self.attention_backend}
        if self.profile == "mxfp4-mxfp4":
            compatible_backends.add("aiter_f4f4")
        if backend_name not in compatible_backends:
            raise ValueError(
                f"Attention A2A profile {self.profile} requires backend {self.attention_backend}, got {backend_name}"
            )
        return self

    def _recipe(self):
        if not self.enabled or self.is_auto:
            raise RuntimeError(f"Attention A2A profile {self.profile!r} has no concrete recipe")
        return _PROFILE_RECIPES[self.profile]

    @property
    def codecs(self) -> tuple[str, str, str]:
        if not self.enabled:
            return ("e4m3", "e4m3", "e4m3")
        return self._recipe()["codecs"]

    @property
    def consumer_codecs(self) -> tuple[str, str, str]:
        if not self.enabled:
            return ("e4m3", "e4m3", "e4m3")
        return self._recipe()["consumer_codecs"]

    @property
    def scale_modes(self) -> tuple[str, str, str]:
        if not self.enabled:
            return ("f32_per_tensor",) * 3
        return self._recipe()["scale_modes"]

    @property
    def attention_backend(self) -> str | None:
        if not self.enabled:
            return None
        if self.is_auto:
            return None
        return self._recipe()["attention_backend"]

    @property
    def v_pack(self) -> str:
        if not self.enabled:
            return "default"
        return self._recipe()["v_pack"]

    @property
    def local_sequence_multiple(self) -> int:
        if not self.enabled:
            return 1
        return self._recipe()["pad_multiple"]

    @property
    def hadamard_placement(self) -> str:
        if not self.enabled:
            return "none"
        if self.hadamard != "auto":
            return self.hadamard
        return self._recipe()["default_hadamard"]


def attention_a2a_profile_help() -> str:
    return ", ".join(ATTENTION_A2A_PROFILES)
