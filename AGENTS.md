# Repository Guidelines

## Project Structure & Module Organization

This is a vLLM-derived Python project with native extensions. Core Python
source lives in `vllm/`, organized by runtime area such as `core/`,
`worker/`, `engine/`, `entrypoints/`, `model_executor/`, and `lora/`. CUDA,
HIP, and C++ extension code is in `csrc/`, with CMake helpers in `cmake/`.
Tests mirror feature areas under `tests/`, and runnable examples live in
`examples/`. Benchmark and experiment scripts are in `benchmarks/` and
`scripts/`; documentation sources are under `docs/`.

## Build, Test, and Development Commands

- `pip install -r requirements-dev.txt`: install development tooling.
- `pip install -e .`: build and install the project in editable mode.
- `bash format.sh`: format changed Python/C++ files and run yapf, mypy,
  codespell, ruff, isort, and clang-format checks.
- `bash format.sh --all`: run formatting/checks across the repository.
- `pytest tests/`: run the full test suite.
- `pytest tests/core/test_scheduler.py`: run a focused test file.
- `python collect_env.py`: capture local CUDA/PyTorch/runtime diagnostics.

## Coding Style & Naming Conventions

Python style is enforced by `pyproject.toml`: 80-character lines, Ruff checks
for pycodestyle, Pyflakes, bugbear, simplify, and logging-format rules;
mypy targets Python 3.8. Use yapf for Python formatting and clang-format for
`csrc/` files. Prefer descriptive snake_case for functions, variables, and test
files; use PascalCase for classes. Keep imports sorted with isort.

## Testing Guidelines

Tests use pytest and follow `test_*.py` naming. Place new tests beside the
feature they cover, for example scheduler tests in `tests/core/` and OpenAI
entrypoint tests in `tests/entrypoints/`. Add regression tests for bug fixes
and include distributed, kernel, or model-specific coverage when changing those
paths. Some tests require GPUs, model downloads, or distributed setup; document
those requirements.

## Commit & Pull Request Guidelines

Recent commit subjects are short, imperative summaries such as `update scripts`
or `add input dataset analysis`; keep subjects concise and specific. Before
opening a PR, run focused tests plus formatting checks relevant to your change.
PRs should include a clear description, linked issue when applicable, test
results, and screenshots or plots for documentation, benchmark, or visualization
changes. Follow `.github/PULL_REQUEST_TEMPLATE.md` when available.

## Security & Configuration Tips

Do not commit model weights, tokens, private datasets, or machine-specific
paths. Keep environment-specific settings in shell scripts or local config, and
prefer documenting required variables over hardcoding them.

## Pre-Coding Thought Process
- **Clarify Assumptions**: Explicitly state assumptions. When in doubt, ask rather than guess.
- **Address Ambiguity**: If a requirement is ambiguous, list potential interpretations instead of choosing one silently.
- **Identify Optimizations**: If a simpler, more effective approach exists, suggest it immediately.
- **Resolve Inconsistencies**: Pause immediately if you detect logical contradictions or conflicts, and request clarification.

## Simplicity First
- **Prioritize Minimalism**: Solve problems with the least amount of code possible. Avoid redundant implementations.
- **Avoid Over-Engineering**: Do not create unnecessary abstraction layers or complex architectures for one-time tasks.
- **Focus on Current Needs**: Do not prematurely add scalability or configurability for "hypothetical future scenarios."
- **Active Refactoring**: If code can be significantly simplified, proactively rewrite and optimize it.
- **Standard of Judgment**: Evaluate code from the perspective of a Senior Engineer; if the solution is overly complex, simplify it.

## Precise Modifications
- **Scope Limitation**: Modify only the code directly related to the current task.
- **Maintain Focus**: Do not "clean up" adjacent code, comments, or formatting unless necessary.
- **Respect Stability**: Do not refactor functional, existing code modules.
- **Style Consistency**: Strictly adhere to the project's existing coding style and conventions.
- **Clean Up**: Remove unused imports or variables directly resulting from your changes.
- **Flag Technical Debt**: If you discover dead code or redundancies, note them for the user but do not delete them without authorization.

## Objective-Driven Execution
- **Define Success**: Establish clear, measurable success criteria before starting.
- **Bug Fixing**: Transform "fixing bugs" into: create a reproduction test case, then debug until the test case passes.
- **Feature Implementation**: Transform "adding validation" into: write test cases for edge cases/invalid inputs and ensure they pass.
- **Refactoring**: Ensure all existing test cases pass after any refactoring.
- **Task Planning**: For complex, multi-step tasks, provide a concise execution plan, including the verification method for each step.