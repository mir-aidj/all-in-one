import os
import sys
import subprocess
from pathlib import Path
from typing import List


def demix(paths: List[Path], demix_dir: Path):
  """Demixes the audio file into its sources."""
  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'htdemucs' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    subprocess.run(
      [
        sys.executable, '-m', 'demucs.separate',
        '-o', demix_dir,
        '-n', 'htdemucs',
        *todos,
      ],
      check=True,
    )

  return demix_paths
