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

## D-03: Relative paths resolve against the YAML file

- **Context:** The specification says suite paths resolve against the YAML file. It gives the
  example `output_dir: ../results` but does not say what that path is relative to.
- **Decision:** Both `quality.suites[].path` and `run.output_dir` resolve against the directory of
  the YAML file. The directory is stored in `Day0Config.base_dir`, which is excluded from every
  hash.
- **Consequence:** `examples/fake.yaml` writes to the repository-level `results/`. The same
  configuration in another directory keeps the same config hash.

## D-04: `gsm8k_200.jsonl` cannot be built in this environment

- **Context:** The specification requires 200 items sampled from the GSM8K test split. Hugging
  Face is not reachable from the build environment. The GitHub repositories that publish the split
  are outside the repository scope granted to the session, so they were not read.
- **Decision:** `scripts/build_suites.py` takes the official `test.jsonl` as an input file
  (`--gsm8k-jsonl`) and does not download it. No GSM8K items are invented or substituted.
- **Consequence:** `examples/llama31-8b-h100.yaml` fails `day0 validate` until the file is built.
  The real-run example is validated in tests with placeholder files in a temporary directory.

## D-05: Shared filler-text module

- **Context:** The perf workload, the fake suites, and the synthetic long-context suite all need
  deterministic filler text from a word list.
- **Decision:** The word list and generators live in `day0/corpus.py`. Each function takes an
  explicit `numpy.random.Generator`. The specification does not list this module, so it is
  recorded here.
- **Consequence:** The generated suites depend only on the seed and this list. Changing the list
  changes every generated suite, so its contents are treated as frozen.

## D-06: Pydantic mypy plugin

- **Context:** `mypy --strict` reports `Missing named argument` errors for model fields that use
  `Field(default, ...)`.
- **Decision:** Enable `pydantic.mypy` with `init_typed` and `init_forbid_extra`.
- **Consequence:** Type checking of configuration models is accurate. Nested default models use
  `default_factory=lambda: Model()` so that mypy infers the type.

## D-07: Strict configuration validation

- **Context:** The specification asks for `extra="forbid"` and for validation rules with clear
  messages.
- **Decision:** Every model forbids unknown keys. Cross-field rules are model validators.
  `reference.enabled: false` is rejected, because the reference always runs.
- **Consequence:** A typo in a YAML key fails with the key name in the message.
