from src.allin1 import analyze, visualize

from pathlib import Path
from src.allin1.utils import mkpath


def test_visualize():
    result = analyze(
        paths=Path(__file__).resolve().parent / "test.mp3",
        keep_byproducts=True,
    )
    visualize(result)


def test_visualize_save():
    result = analyze(
        paths=Path(__file__).resolve().parent / "test.mp3",
        keep_byproducts=True,
    )
    visualize(
        result,
        out_dir="./viz",
    )
    assert mkpath("./viz/test.pdf").is_file()
