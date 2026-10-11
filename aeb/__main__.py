"""`python -m aeb` entry point -- same CLI as the `mono-aeb` console script."""

from aeb.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
