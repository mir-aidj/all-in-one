# All-In-One Music Structure Analysis Model

[![PyPI - Version](https://img.shields.io/pypi/v/allin1.svg)](https://pypi.org/project/haha)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/allin1.svg)](https://pypi.org/project/allin1)

-----

> NOTE: This is a work in progress

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Advanced Usage for Research](#advanced-usage-for-research)
- [Citation](#citation)

## Installation

### 1. Install PyTorch

Visit [PyTorch](https://pytorch.org/) and install the appropriate version for your system.

### 2. Install NATTEN (For Linux and Windows only, not required for macOS)

#### Linux
Visit [NATTEN website](https://www.shi-labs.com/natten/) and download the appropriate version for your system.

#### macOS
No need to install NATTEN, it will be installed automatically when installing `allin1`.

#### Windows
Build NATTEN from source:
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

```shell
allin1 your_audio_file1.wav your_audio_file2.mp3
```
The result will be saved in `./structures:
```shell
./structures
└── your_audio_file1.json
└── your_audio_file2.json
```
And a JSON analysis result has:
```json
{
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

### Python

```python
import allinone

# You can analyze a single file:
result = allinone.analyze('your_audio_file.wav')

# Or multiple files:
results = allinone.analyze(['your_audio_file1.wav', 'your_audio_file2.mp3'])
```
A result is a dataclass instance containing:
```python
AnalysisResult(
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

## Advanced Usage for Research

TODO

## Available Models

TODO

## Citation

```bibtex
@inproceedings{taejun2023allinone,
  title={All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio},
  author={Kim, Taejun and Nam, Juhan},
  booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023}
}
```