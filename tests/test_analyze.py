import allin1

from pathlib import Path

CWD = Path(__file__).resolve().parent


def test_analyze():
  allin1.analyze(CWD / 'test.mp3')
