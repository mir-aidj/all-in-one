from src.allin1 import AnalysisResult, sonify

from pathlib import Path

CWD = Path(__file__).resolve().parent


def test_sonify():
    result = AnalysisResult.from_json(CWD / "data" / "test.json")
    sonify(result)


def test_sonify_save():
    result = AnalysisResult.from_json(CWD / "data" / "test.json")
    sonify(result, out_dir="./sonif")
