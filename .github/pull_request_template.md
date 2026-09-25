## Summary

<!-- What changed, and why? Link the relevant issue with "Closes #..." when applicable. -->

## Validation

Automated lint, formatting, CPU unit, and CPU integration checks run in PR CI.

### Accelerator tests

- [ ] NVIDIA tests passed
- [ ] AMD/ROCm tests passed
- [ ] Multi-GPU tests passed
- [ ] Not applicable (explain below)

Hardware, command, and result:

<!-- Example: 2x MI300X; python -m pytest tests -m "rocm and multi_gpu"; 42 passed -->

### End-to-end tests

- [ ] Appropriate model/pipeline E2E tests passed, including output-quality review
- [ ] Performance and memory were checked for kernel, distributed, or inference-path changes
- [ ] Not applicable (explain below)

Model, command, result, and any relevant performance or quality comparison:

## Notes

<!-- Explain unchecked items, known limitations, or follow-up work. -->
