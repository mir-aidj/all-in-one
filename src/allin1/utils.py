import re

from pathlib import Path
from .typings import PathLike, AnalysisResult


def compact_json_number_array(json_str: str):
  """Compact numbers (including floats) in JSON arrays to be on the same line."""
  return re.sub(
    r'(\[\n(?:\s*\d+(\.\d+)?,\n)+\s*\d+(\.\d+)?\n\s*\])',
    lambda m: m.group(1).replace('\n', '').replace(' ', ''),
    json_str
  )


def mkpath(path: PathLike):
  return Path(path).expanduser().resolve()


def load_result(
  path: PathLike,
  load_activations: bool = True,
  load_embeddings: bool = True,
) -> AnalysisResult:
  path = mkpath(path)
  result = AnalysisResult.from_json(
    path,
    load_activations=load_activations,
    load_embeddings=load_embeddings,
  )
  return result
