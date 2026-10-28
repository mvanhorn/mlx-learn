"""Repeat the general scaled rank benchmark at C=10, with and without weights.

This control uses the unchanged primary rank benchmark and explicitly records
C=10 in each case. Its additional source is included in both integrity hashes.
The timed fits, tight centered float64 reference, null-space checks, warnings,
and route checks are identical to the primary protocol.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from . import rank_safe_logistic as rank


def main():
    original_cases, original_sources, original_doc = rank.CASES, rank.sources, rank.__doc__
    rank.CASES = tuple(replace(case, name=case.name.replace("_4096", "_C10_4096"), C=10.0)
                       for case in original_cases if case.scaled)
    rank.__doc__ = __doc__

    def sources(extra=()):
        return original_sources((*extra, Path(__file__).resolve()))

    rank.sources = sources
    try:
        return rank.main()
    finally:
        rank.CASES, rank.sources, rank.__doc__ = original_cases, original_sources, original_doc


if __name__ == "__main__":
    raise SystemExit(main())
