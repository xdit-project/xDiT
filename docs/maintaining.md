# Build and update the docs

## Preview locally

From the repository root:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install --upgrade pip
.venv-docs/bin/python -m pip install --group docs
.venv-docs/bin/python -m mkdocs serve
```

Open the URL printed by MkDocs. To check the site without starting a server,
run `.venv-docs/bin/python -m mkdocs build --strict`. The output goes to `site/`.
Neither command needs xDiT, model weights, or accelerators.

## Lint Markdown

Install `pre-commit`, then run `pre-commit run pymarkdown --all-files` from the
repository root. CI checks all tracked Markdown, including archived pages.
Run `pre-commit install` to enable the hooks before local commits.

The rules in `.pymarkdown.json` allow HTML and long lines for anchors, diagrams,
URLs, and generated tables. Pages can start with a logo or skill metadata.
Other default checks remain enabled, including headings, lists, fences, and links.

## Keep pages current

Follow the [documentation skill](https://github.com/xdit-project/xDiT/blob/main/.github/skills/documentation/SKILL.md)
for source checks, writing, and diagram review.

Keep option details in the parser and model declarations. Regenerate the
references in an environment with xDiT installed:

```bash
python -m scripts.docs.generate_cli_reference
python -m scripts.docs.generate_model_reference
```

Regenerate after changing flags, model aliases, capabilities, or cache presets.
Commit generated pages with their source changes; CI checks for drift.
Update `scripts/docs/model_reference_io.json` from each adapter's input/output handling.
The generator rejects missing or retired adapter entries.
The attention backend table is maintained manually; check it against the enum
and backend specifications.

When updating MkDocs, check the licensed `docs/theme/js/darkmode.js` override.
It sets SVG colors and handles disabled syntax highlighting.
Verify Auto follows system changes and manual choices persist after reload.

## Publish with GitHub Pages

The workflow builds the site and deploys its artifact through GitHub Actions.
**It does not need a `gh-pages` branch or a separate deployment token.**

Publication is restricted to the default branch in `xdit-project/xDiT`, currently
`main`. Pull requests build the site without publishing.

1. A repository maintainer enables **Settings → Pages → Build and deployment →
   Source: GitHub Actions**.
2. Merge the reviewed documentation and workflow into `main`. A successful
   workflow publishes at <https://xdit-project.github.io/xDiT/>.

After setup, pushes to `main` update the site automatically. Use workflow dispatch
on the default branch to redeploy it.

See [GitHub's custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
