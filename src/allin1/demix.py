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
        print(f'=> {path} is already demixed.')
        continue
    todos.append(path)

  if todos:
    print(f'=> Start demixing {len(todos)} tracks.')
    subprocess.run([
      sys.executable, '-m', 'demucs.separate',
      '-o', demix_dir,
      '-n', 'htdemucs',
      ' '.join([str(p) for p in todos]),
    ])

  return demix_paths
