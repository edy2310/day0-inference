# Decision log

Each entry records a choice that the implementation specification did not dictate.
Format: context, decision, consequence.

## D-01: Repository root is the `day0/` project root

- **Context:** The specification lays out the tree under a top-level `day0/` directory.
- **Decision:** The git repository root is the project root. `pyproject.toml`, `README.md`,
  `day0/` (package), `suites/`, `examples/`, `scripts/`, `tests/`, and `docs/` live at the top
  level of the repository.
- **Consequence:** Paths in the specification map one-to-one to repository paths.

## D-02: Development environment

- **Context:** The specification requires Python 3.11+ and tests on 3.11 and 3.12.
- **Decision:** Local development uses a `.venv` created from Python 3.11. CI runs the matrix
  3.11 and 3.12.
- **Consequence:** `.venv/` is git-ignored. Ruff line length is 100.
