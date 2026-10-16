# Contributing

1. Fork, clone, run `python -m unittest discover -s tests -t .` first.
2. Keep the project zero-dependency: no numpy/pytorch in the core path.
3. Add a benchmark to `bench/` for any new mechanism, with a baseline and a ceiling.
4. Run `python -m sparkforge.doctor` before submitting.
