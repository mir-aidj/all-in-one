import allin1

from pathlib import Path
from allin1.utils import mkpath


def test_visualize():
  result = allin1.analyze(
    paths=Path(__file__).resolve().parent / 'test.mp3',
    keep_byproducts=True,
  )
  allin1.visualize(result)


def test_visualize_save():
  result = allin1.analyze(
    paths=Path(__file__).resolve().parent / 'test.mp3',
    keep_byproducts=True,
  )
  allin1.visualize(
    result,
    out_dir='./viz',
  )
  assert mkpath('./viz/test.pdf').is_file()
