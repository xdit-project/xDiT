## Summary

<!-- What changed, and why? Link the relevant issue with "Closes #..." when applicable. -->

## Validation

Automated lint, formatting, CPU unit, and CPU integration checks run in PR CI. They do not cover accelerator behavior.

Record the hardware, environment, command, and passed/skipped/deselected counts for every suite you ran. Focused tests do not replace the applicable CPU suites or the accelerator selectors the available hardware supports.

### Accelerator tests

- [ ] NVIDIA tests passed
- [ ] AMD/ROCm tests passed
- [ ] Multi-GPU tests passed
- [ ] Not applicable (explain below)

Hardware, environment, command, and result:

<!-- Example: 8x MI300X; python -m pytest tests/integration/accelerator -m "accelerator and not nvidia"; X passed, Y deselected -->

### End-to-end tests

- [ ] Appropriate model/pipeline E2E tests passed, including output-quality review
- [ ] Performance and memory were checked for kernel, distributed, or inference-path changes
- [ ] Not applicable (explain below)

Model, command, result, and any relevant performance or quality comparison:

## Notes

<!-- Explain unchecked items, known limitations, or follow-up work. -->
