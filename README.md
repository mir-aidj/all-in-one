# All-In-One Music Structure Analyzer
[![PyPI - Version](https://img.shields.io/pypi/v/allin1.svg)](https://pypi.org/project/allin1)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/allin1.svg)](https://pypi.org/project/allin1)

This package provides models for music structure analysis, predicting:
1. Tempo (BPM)
2. Beats
3. Downbeats
4. Section boundaries
5. Section labels (e.g., intro, verse, chorus, bridge, outro)


-----


**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Available Models](#available-models)
- [Speed](#speed)
- [Advanced Usage for Research](#advanced-usage-for-research)
- [Concerning MP3 Files](#concerning-mp3-files)
- [Citation](#citation)

## Installation

### 1. Install PyTorch

Visit [PyTorch](https://pytorch.org/) and install the appropriate version for your system.

### 2. Install NATTEN (Required for Linux and Windows; macOS will auto-install)
* **Linux**: Download from [NATTEN website](https://www.shi-labs.com/natten/)
* **macOS**: Auto-installs with `allin1`.
* **Windows**: Build from source:
```shell
pip install ninja # Recommended, not required
git clone https://github.com/SHI-Labs/NATTEN
cd NATTEN
make
```

### 3. Install the package

```shell
pip install git+https://github.com/CPJKU/madmom  # install the latest madmom directly from GitHub
pip install allin1  # install this package
```

### 4. (Optional) Install FFmpeg for MP3 support

For ubuntu:

```shell
sudo apt install ffmpeg
```

For macOS:

```shell
brew install ffmpeg
```

## Usage

### CLI
Run:
```shell
allin1 your_audio_file1.wav your_audio_file2.mp3
```
Results are saved in `./structures:
```shell
./structures
└── your_audio_file1.json
└── your_audio_file2.json
```
And a JSON analysis result has:
```json
{
  "path": "/path/to/your_audio_file.wav",
  "bpm": 100,
  "beats": [ 0.33, 0.75, 1.14, ... ],
  "downbeats": [ 0.33, 1.94, 3.53, ... ],
  "beat_positions": [ 1, 2, 3, 4, 1, 2, 3, 4, 1, ... ],
  "segments": [
    {
      "start": 0.0,
      "end": 0.33,
      "label": "start"
    },
    {
      "start": 0.33,
      "end": 13.13,
      "label": "intro"
    },
    {
      "start": 13.13,
      "end": 37.53,
      "label": "chorus"
    },
    {
      "start": 37.53,
      "end": 51.53,
      "label": "verse"
    },
    ...
  ]
}
```
Available options:
```shell
$ allin1 --help

usage: allin1 [-h] [-a] [-e] [-o OUT_DIR] [-m MODEL] [-d DEVICE] [-k] [--demix-dir DEMIX_DIR] [--spec-dir SPEC_DIR] paths [paths ...]

positional arguments:
  paths                 Path to tracks

options:
  -h, --help            show this help message and exit
  -a, --activ           Save frame-level raw activations from sigmoid and softmax (default: False)
  -e, --embed           Save frame-level embeddings (default: False)
  -o OUT_DIR, --out-dir OUT_DIR
                        Path to a directory to store analysis results (default: ./structures)
  -m MODEL, --model MODEL
                        Name of the pretrained model to use (default: harmonix-all)
  -d DEVICE, --device DEVICE
                        Device to use (default: cuda if available else cpu)
  -k, --keep-byproducts
                        Keep demixed audio files and spectrograms (default: False)
  --demix-dir DEMIX_DIR
                        Path to a directory to store demixed tracks (default: ./demixed)
  --spec-dir SPEC_DIR   Path to a directory to store spectrograms (default: ./spectrograms)
```

### Python

```python
import allin1

# You can analyze a single file:
result = allin1.analyze('your_audio_file.wav')

# Or multiple files:
results = allin1.analyze(['your_audio_file1.wav', 'your_audio_file2.mp3'])
```
A result is a dataclass instance containing:
```python
AnalysisResult(
  path='/path/to/your_audio_file.wav', 
  bpm=100,
  beats=[0.33, 0.75, 1.14, ...],
  beat_positions=[1, 2, 3, 4, 1, 2, 3, 4, 1, ...],
  downbeats=[0.33, 1.94, 3.53, ...], 
  segments=[
    Segment(start=0.0, end=0.33, label='start'), 
    Segment(start=0.33, end=13.13, label='intro'), 
    Segment(start=13.13, end=37.53, label='chorus'), 
    Segment(start=37.53, end=51.53, label='verse'), 
    Segment(start=51.53, end=64.34, label='verse'), 
    Segment(start=64.34, end=89.93, label='chorus'), 
    Segment(start=89.93, end=105.93, label='bridge'), 
    Segment(start=105.93, end=134.74, label='chorus'), 
    Segment(start=134.74, end=153.95, label='chorus'), 
    Segment(start=153.95, end=154.67, label='end'),
  ]),
```
Unlike CLI, it does not save the results to disk by default. You can save them as follows:
```python
result = allin1.analyze(
  'your_audio_file.wav',
  out_dir='./structures',  # None by default
)
``` 
The Python API `allin1.analyze()` offers the same options as the CLI:
```python
def analyze(
  paths: PathLike | List[PathLike],
  out_dir: PathLike = None,
  model: str = 'harmonix-all',
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demixed',
  spec_dir: PathLike = './spectrograms',
  keep_byproducts: bool = False,
): ...
```


## Available Models
The models are trained on the [Harmonix Set](https://github.com/urinieto/harmonixset) with 8-fold cross-validation.
For more details, please refer to the [paper](http://arxiv.org/abs/2307.16425).
* `harmonix-all`: (Default) An ensemble model averaging the predictions of 8 models trained on each fold.
* `harmonix-foldN`: A model trained on fold N (0~7). For example, `harmonix-fold0` is trained on fold 0.

By default, the `harmonix-all` model is used. To use a different model, use the `--model` option:
```shell
allin1 --model harmonix-fold0 your_audio_file.wav
```


## Speed
With an RTX 4090 GPU and Intel i9-10940X CPU (14 cores, 28 threads, 3.30 GHz),
the `harmonix-all` model processed 10 songs (33 minutes) in 73 seconds.


## Advanced Usage for Research
This package provides researchers with advanced options to extract **frame-level raw activations and embeddings** 
without post-processing. These have a resolution of 100 FPS, equivalent to 0.01 seconds per frame.

### CLI

#### Activations
The `--activ` option also saves frame-level raw activations from sigmoid and softmax:
```shell
$ allin1 --activ your_audio_file.wav
```
You can find the activations in the `.npz` file:
```shell
./structures
└── your_audio_file1.json
└── your_audio_file1.activ.npz
```
To load the activations in Python:
```python
>>> import numpy as np
>>> activ = np.load('./structures/your_audio_file1.activ.npz')
>>> activ.files
['beat', 'downbeat', 'segment', 'label']
>>> beat_activations = activ['beat']
>>> downbeat_activations = activ['downbeat']
>>> segment_boundary_activations = activ['segment']
>>> segment_label_activations = activ['label']
```
Details of the activations are as follows:
* `beat`: Raw activations from the **sigmoid** layer for **beat tracking** (shape: `[time_steps]`)
* `downbeat`: Raw activations from the **sigmoid** layer for **downbeat tracking** (shape: `[time_steps]`)
* `segment`: Raw activations from the **sigmoid** layer for **segment boundary detection** (shape: `[time_steps]`)
* `label`: Raw activations from the **softmax** layer for **segment labeling** (shape: `[label_class=10, time_steps]`)

You can access the label names as follows:
```python
>>> allin1.HARMONIX_LABELS
['start',
 'end',
 'intro',
 'outro',
 'break',
 'bridge',
 'inst',
 'solo',
 'verse',
 'chorus']
```


#### Embeddings
This package also provides an option to extract raw embeddings from the model.
```shell
$ allin1 --embed your_audio_file.wav
```
You can find the embeddings in the `.npy` file:
```shell
./structures
└── your_audio_file1.json
└── your_audio_file1.embed.npy
```
To load the embeddings in Python:
```python
>>> import numpy as np
>>> embed = np.load('your_audio_file1.embed.npy')
```
Each model embeds for every source-separated stem per time step, 
resulting in embeddings shaped as `[stems=4, time_steps, embedding_size=24]`:
1. The number of source-separated stems (the order is bass, drums, other, vocals).
2. The number of time steps (frames). The time step is 0.01 seconds (100 FPS).
3. The embedding size of 24.

Using the `--embed` option with the `harmonix-all` ensemble model will stack the embeddings, 
saving them with the shape `[stems=4, time_steps, embedding_size=24, models=8]`.

### Python
The Python API `allin1.analyze()` offers the same options as the CLI:
```python
>>> allin1.analyze(
      paths='your_audio_file.wav',
      include_activations=True,
      include_embeddings=True,
    )

AnalysisResult(
  path='/path/to/your_audio_file.wav', 
  bpm=100, 
  beats=[...],
  downbeats=[...],
  segments=[...],
  activations={
    'beat': array(...), 
    'downbeat': array(...), 
    'segment': array(...), 
    'label': array(...)
  }, 
  embeddings=array(...),
)
```

## Concerning MP3 Files
Due to variations in decoders, MP3 files can have slight offset differences.
I recommend you to first convert your audio files to WAV format using FFmpeg (as shown below), 
and use the WAV files for all your data processing pipelines.
```shell
ffmpeg -i your_audio_file.mp3 your_audio_file.wav
```
In this package, audio files are read using [Demucs](https://github.com/facebookresearch/demucs).
To my understanding, Demucs converts MP3 files to WAV using FFmpeg before reading them.
However, using a different MP3 decoder can yield different offsets. 
I've observed variations of about 20~40ms, which is problematic for tasks requiring precise timing like beat tracking, 
where the conventional tolerance is just 70ms. 
Hence, I advise standardizing inputs to the WAV format for all data processing, 
ensuring straightforward decoding.


## Citation
If you use this package for your research, please cite the following paper:
```bibtex
@inproceedings{taejun2023allinone,
  title={All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio},
  author={Kim, Taejun and Nam, Juhan},
  booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023}
}
```