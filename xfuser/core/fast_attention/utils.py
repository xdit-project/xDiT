# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/thu-nics/DiTFastAttn/blob/main/dit_fast_attention.py
# Copyright (c) 2024 NICS-EFC Lab of Tsinghua University.
"""Read and write the DiTFastAttn compression config file.

The calibration that selects the methods is in
``xfuser.model_executor.fast_attention_calibration``.
"""

import json
import os

from .attn_layer import FastAttnMethod


def save_config_file(step_methods, file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    format_data = {
        f"block{blocki}": {f"step{stepi}": method.name for stepi, method in enumerate(methods)}
        for blocki, methods in enumerate(step_methods)
    }
    with open(file_path, "w") as file:
        json.dump(format_data, file, indent=2)


def load_config_file(file_path):
    with open(file_path, "r") as file:
        format_data = json.load(file)
    steps_methods = [
        [FastAttnMethod[method] for method in format_method.values()] for format_method in format_data.values()
    ]
    return steps_methods
