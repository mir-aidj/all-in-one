import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import librosa.feature
import demucs.separate

from typing import Union, List, Mapping

from .typings import AnalysisResult, PathLike
from .utils import mkpath

HARMONIX_COLORS = {
  'start': 'black',
  'end': 'black',
  'intro': 1,
  'outro': 1,
  'break': 2,
  'bridge': 2,
  'inst': 3,
  'solo': 3,
  'verse': 4,
  'chorus': 5,
}


def visualize(
  results: Union[AnalysisResult, List[AnalysisResult]],
  out_dir: PathLike = None,
) -> Union[plt.Figure, List[plt.Figure]]:
  return_list = True
  if not isinstance(results, list):
    return_list = False
    results = [results]

  figs = [_plot(r) for r in results]

  # Save figures to out_dir
  if out_dir is not None:
    out_dir = mkpath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result, fig in zip(results, figs):
      fig.savefig(out_dir / f'{result.path.stem}.pdf', bbox_inches='tight')

  if not return_list:
    return figs[0]
  return figs


def _plot(
  result: AnalysisResult,
  colors: Mapping[str, int] = None,
  color_map: str = 'viridis',
):
  if colors is None:
    colors = HARMONIX_COLORS

  sr = 44100
  y = demucs.separate.load_track(result.path, 1, sr)[0].numpy()
  # y, sr = librosa.load(result.path, sr=None, mono=True)
  rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=1024)[0]

  fig = plt.figure(figsize=(12, 2))
  gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1])

  ax0.plot(rms, color='black', linewidth=1)
  ax0.set_xlim(0, len(rms) - 1)
  ax0.set_ylim(0, None)
  ax0.set_ylabel('RMS')
  ax0.set_xticks([])

  cmap = plt.get_cmap(color_map)
  max_color = max(c for c in colors.values() if not isinstance(c, str))
  min_color = min(c for c in colors.values() if not isinstance(c, str))
  for segment in result.segments:
    color = colors[segment.label]
    if not isinstance(color, str):
      color = (color - min_color) / (max_color - min_color)
      color = cmap(color)

    ax1.axvspan(segment.start, segment.end, color=color)
    ax1.axvline(segment.start, color='black', linewidth=1)
    if segment.label not in ['start', 'end']:
      ax1.text(
        (segment.end - segment.start) / 2 + segment.start,
        0.5,
        segment.label,
        color=_get_text_color(color),
        fontsize=12,
        weight='bold',
        horizontalalignment='center',
        verticalalignment='center',
      )
  ax1.set_xlim(0, result.segments[-1].end)
  tick_segments = [s for s in result.segments if s.label != 'start']
  ax1.set_xticks(
    [s.start for s in tick_segments],
    [f'{round(s.start // 60)}:{round(s.start % 60):02}' for s in tick_segments],
  )
  ax1.set_xlabel('Time (min:sec)')
  ax1.set_yticks([])

  # set title
  ax0.set_title(result.path.name)
  fig.tight_layout()
  fig.subplots_adjust(hspace=0)

  return fig


def _get_text_color(bg_color):
  bg_color = mcolors.to_rgb(bg_color)
  luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]

  if luminance > 0.5:
    return 'black'
  else:
    return 'white'
