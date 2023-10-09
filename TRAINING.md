# Training All-In-One Music Structure Analaysis Models

## Install Dependencies

```shell
pip install allin1[train]
```

## Setup Weights & Biases

This package uses [Weights & Biases](https://wandb.ai/site) to log and track training.

1. Sign up [here](https://wandb.ai/site).
2. Login from terminal:

```shell
wandb login
```

## Prepare Data (Harmonix Set)

Make a directory for Harmonix Set:

```shell
mkdir -p data/harmonix
```

You should refer to [Harmonix Set](https://github.com/urinieto/harmonixset)
and collect the audio files by yourself.
Then, place the audio files, metadata, and annotations in the following structure:

```shell
data/harmonix
|-- metadata.csv
|-- beats
|   |-- 0001_12step.txt
|   |-- ...
|   `-- 1001_yourloveismydrugdave.txt
|-- segments
|   |-- 0001_12step.txt
|   |-- ...
|   `-- 1001_yourloveismydrugdave.txt
`-- tracks
|   |-- 0001_12step.mp3
|   |-- ...
|   `-- 1001_yourloveismydrugdave.mp3
```

## Preprocess Data

The following command will preprocess the data and save source-separated spectrograms in `data/harmonix/features`:

```shell
allin1-preprocess
```

# Train

The following command will train the model and upload the model and training logs to Weights & Biases:

```shell
allin1-train
```

Then, on your terminal, you will see the link where you can monitor the training:

```shell
wandb: 🚀 View run at https://wandb.ai/YOUR_ID/YOUR_PROJECT/runs/xxxxxxxx
````

You can put options, for example, to specify the fold to train:

```shell
allin1-train fold=2
```

To see all the options, check `src/allin1/config.py` or run:

```shell
allin1-train --help
```

You can also run multiple training jobs in parallel using [Weights & Biases sweep](https://docs.wandb.ai/guides/sweeps).
The example below is a sweep configuration file `sweep.yaml` for training all folds:

```yaml
project: all-in-one
program: allin1-train
method: grid
description: >
  Train All-In-One Music Structure Analysis Models for all folds.

metric:
  name: test/loss
  goal: minimize

parameters:
  fold:
    values: [ 0, 1, 2, 3, 4, 5, 6, 7 ]

command:
  - ${env}
  - ${program}
  - ${args_no_hyphens}
```

You can run the sweep with the following command:

```shell
wandb sweep sweep.yaml
```

And on each GPU, run the following command to start training:

```shell
CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>
CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID>
...
```
