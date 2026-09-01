#!/usr/bin/env python3
"""Opt-in calibration-audit runtime measurement; timings are never snapshotted."""

from __future__ import annotations

import time

from generate_calibration_validation import generate


def main() -> None:
    started = time.perf_counter()
    generate()
    elapsed = time.perf_counter() - started
    print(f"bounded Step 11 audit report: {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()
