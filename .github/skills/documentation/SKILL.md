---
name: documentation
description: Update xDiT docs, examples, navigation, and diagrams against current source.
---

# xDiT documentation

Use clear technical English inspired by [ASD-STE100](https://www.asd-ste100.org/),
without claiming formal compliance.

- Verify flags and features against the parser, adapters, backend specs, and execution.
  Parser acceptance does not establish support.
- State inputs, defaults, dependencies, hardware, and restrictions. Check each example's
  model, flags, and process count. Label legacy interfaces.
- Regenerate CLI and model tables with `scripts/docs/`; do not edit generated pages.
  Check input/output metadata and the manual attention table against source.
- Preserve external sources' titles and verify dates. Keep descriptions factual.
  Distinguish published images from local container builds.
- Use short, active sentences, consistent terms, and exact identifiers.
  Put prerequisites first and link to detail instead of repeating it.
- Organize pages around user tasks. Keep general guidance vendor-neutral.
  Exclude edit history and review conversations.
- Archive retired guides and measurements in `docs/archive/`.
  Update paired translations and check their links.
- Preserve the official logo. Render accessible SVGs in both themes;
  check execution, text alignment, padding, and arrows.
- Run `pre-commit run pymarkdown --all-files` and `python -m mkdocs build --strict`.
  Check links, search, archives, and desktop/mobile layouts. Keep flags intact and
  tables scrolling within the page.
