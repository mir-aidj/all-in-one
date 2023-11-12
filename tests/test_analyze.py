import pytest
import time
import warnings
from pathlib import Path

import numpy as np

from src.allin1.analyze import analyze

CWD = Path(__file__).resolve().parent


def test_analyze():
    start = time.time()
    analyze(CWD / "data" / "test.wav")
    end = time.time()
    timing = end - start

    warnings.warn(UserWarning(f"`analyze` on a 3:44 minutes-long WAV took {timing}"))


def test__short_analyze():
    from tests.expected.analyze import expected_short

    start = time.time()
    results = analyze(CWD / "data" / "test-chunk.wav", out_dir=CWD / "data")
    end = time.time()
    timing = end - start

    warnings.warn(UserWarning(f"`analyze` on a 35 secs-long WAV took {timing}"))

    assert str(results.path.resolve()).endswith("tests/data/test-chunk.wav")
    assert results.bpm == expected_short.bpm

    assert np.allclose(results.beats[0:6], expected_short.beats[0:6], atol=1e-01)
    assert np.allclose(results.downbeats[0:4], expected_short.downbeats[0:4], atol=1e-01)
    assert np.allclose(results.beat_positions[0:5], expected_short.beat_positions[0:5], atol=1e-01)
    assert np.allclose(
        [[s.start, s.end] for s in results.segments],
        [[s.start, s.end] for s in expected_short.segments],
        atol=1e-01,
    )
