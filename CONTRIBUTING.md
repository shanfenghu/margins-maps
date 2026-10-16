# Contributing

Use Python 3.11 or newer and install the development dependencies with
`python -m pip install -e '.[dev]'`.

Before opening a pull request, run `make audit`. Add deterministic tests for
behavioural changes, keep generated stress tests on explicit seeds, and avoid
committing local environments or caches. Changes to scientific inputs must also
update their expected digests and provenance notes.

Keep domain assumptions explicit in names and docstrings. Prefer independent
certificate replay over checks that repeat the same implementation path.
