import allin1

from pathlib import Path

CWD = Path(__file__).resolve().parent


def test_sonify():
  result = allin1.AnalysisResult.from_json(CWD / 'test.json')
  allin1.sonify(result)


def test_sonify_save():
  result = allin1.AnalysisResult.from_json(CWD / 'test.json')
  allin1.sonify(
    result,
    out_dir='./sonif'
  )
