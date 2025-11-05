# expert-paper-extract

Repository of Expert-Grounded Prompt Engineering for extracting lattice constants of high entropy alloys from scientific publications using large language models.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (installs packages and manages the project environment)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/expert-paper-extract.git
cd expert-paper-extract
```

### 2. Install dependencies

The project uses `uv` for dependency management. Sync the environment and install the package in editable mode:

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root (or update the existing one) with the credentials your workflows require:

```bash
ANTHROPIC_API_KEY=your_key
OpenAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

Never commit real keys to version control; keep `.env` listed in `.gitignore`.

## Running the scripts

- The Jupyter notebooks (`textgrad_training.ipynb`, `textgrad_output_evaluation.ipynb`, `json_to_dataframe.ipynb`) contain prompt optimization and output evaluations as described in the paper. Launch them with `uv run jupyter notebook` or `uv run jupyter lab`.
- `large_scale_extraction.py` is the python script that drives large-scale PDF data extraction. Start it with `uv run python large_scale_extraction.py`.

## Example outputs and utilities

- `example_outputs/optimized_prompt_Claude35(f)_Claude35(b).md` documents an optimized prompt from previous experiments, while the accompanying `.log` file records the iterative TextGrad training run. Review and redact before sharing externally.
- `utilities/Check_Composition_Consistency.py` normalizes alloy compositions, cross-checks nominal vs measured values, and augments CSV datasets with similarity metrics.
- `utilities/pdf_token_counter.py` estimates document length (pages, words, tokens) across multiple extraction backends and can export summaries to CSV.

## Contributing

1. Create a new branch for your changes.
2. Run the relevant scripts or notebooks to verify behaviour.
3. Submit a pull request with a clear description of the updates.
