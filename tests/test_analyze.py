from src.allin1.analyze import analyze

import time
from pathlib import Path

CWD = Path(__file__).resolve().parent

import warnings




def test_analyze():
    start = time.time()
    analyze(CWD / "data" / "test.wav")
    end = time.time()
    timing = end - start 

    warnings.warn(
        UserWarning(
            f"`analyze` on a 3:44 minutes long song took {timing}"))


def test_analyze_short():
    from tests.expected.analyze import expected_short

    start = time.time()
    results = analyze(CWD / "data" / "test-chunk.wav")
    end = time.time()
    timing = end - start 

    warnings.warn(
        UserWarning(
            f"`analyze` on a 35 secs long song took {timing}"))

    assert str(results.path.resolve()).endswith("tests/data/test-chunk.wav")
    assert results.bpm == expected_short.bpm
    assert results.beats[0:6] == expected_short.beats[0:6]

    assert results.downbeats[0:4] == expected_short.downbeats[0:4]
    assert results.beat_positions[0:5] == expected_short.beat_positions[0:5]
    assert results.segments == expected_short.segments
