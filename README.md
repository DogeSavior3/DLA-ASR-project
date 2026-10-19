# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This project implements an automatic speech recognition (ASR) system with PyTorch, based on DeepSpeech2 with CTC decoding. It was developed for the [HSE DLA course](https://github.com/markovka17/dla) and uses the [ASR Project Template](https://github.com/Blinorot/pytorch_project_template/tree/example/asr) as its foundation.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

1. Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/). For example, on Linux or macOS:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Restart your terminal if `uv` is not yet available on your `PATH`.

   On macOS, installing with Homebrew is recommended:

   ```bash
   brew install uv
   ```

2. Clone the repository and install the locked dependencies:

   ```bash
   git clone https://github.com/DogeSavior3/DLA-ASR-project.git
   cd DLA-ASR-project
   uv sync --locked --python 3.13
   ```

   uv creates a `.venv` environment and downloads Python 3.13 if needed. Dependencies are defined in `pyproject.toml` and pinned in `uv.lock`.

3. Activate the environment to use the commands below:

   ```bash
   source .venv/bin/activate
   ```

   Alternatively, run commands without activation using `uv run`, for example `uv run python train.py -cn=deepspeech2_baseline`.

4. (Optional, for development) Install the pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

A demo notebook is available in the root of the project.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
