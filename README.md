# tiny-trons

An educational implementation of GPT-style transformer models from scratch using PyTorch. Trained on TinyShakespeare for character-level language modeling.

## Features

- Character-level transformer language model (SmolTRF)
- Simple Bigram baseline model
- INT8 quantization experiments
- Automatic GPU detection (CUDA/ROCm/MPS)

## Requirements

- Python >= 3.12
- [uv](https://astral.sh/uv) package manager

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with dev dependencies (for testing/linting)
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
```

### WSL + AMD GPU (ROCm)

After `uv sync`, replace PyTorch with ROCm version:

```bash
uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

## Usage

### Training

```bash
# Run the main training script
python src/lm/train.py
```

The training script will:
- Load the TinyShakespeare corpus (~1.1MB)
- Train a Bigram language model for 3000 iterations
- Log training/validation loss every 200 steps
- Generate sample text during training

### Configuration

Hyperparameters are configured in `src/lm/utils.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_sz` | 32 | Batch size |
| `block_sz` | 8 | Context window size |
| `max_iters` | 3000 | Training iterations |
| `eval_iters` | 200 | Steps between evaluations |
| `lr` | 3e-4 | Learning rate |
| `device` | auto | Automatically detects CUDA/CPU |

## Project Structure

```
tiny-trons/
├── src/
│   ├── main.py              # Alternative entry point
│   ├── tokenizer.py         # Character tokenizer
│   ├── lm/                   # Language model components
│   │   ├── train.py          # Main training script
│   │   ├── bigram.py         # Bigram language model
│   │   └── utils.py          # Hyperparameters & settings
│   ├── data/
│   │   ├── utils.py          # Dataset & Dataloader
│   │   └── tinyshakespeare.txt
│   ├── quant/
│   │   └── int8.py           # INT8 quantization experiments
│   ├── ft/                   # Fine-tuning utilities
│   └── prompt/               # Prompt optimization experiments
├── tests/                    # Test suite
├── pyproject.toml
└── CLAUDE.md                 # Detailed architecture docs
```

## Development

### Linting & Formatting

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

### Type Checking

```bash
pyright src/ tests/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tokenize.py

# Run with pattern matching
pytest -k test_tokenizer

# Verbose output
pytest -v
```

### Pre-commit Hooks

Pre-commit hooks run automatically:
- **On commit**: ruff linter/formatter, trailing whitespace, nbstripout
- **On push**: pyright type checking, pytest

```bash
# Run manually on all files
pre-commit run --all-files
```

## Architecture

### BigramLM (`src/lm/bigram.py`)

Simple baseline model using embedding lookup. Each token predicts the next based only on its embedding.

### SmolTRF (`src/lmold/trf.py`)

Full transformer implementation with:
- **Multi-head self-attention** with causal masking
- **Feed-forward network** (4x expansion with ReLU)
- **Pre-norm architecture** (LayerNorm before attention/FFN)
- **Learned positional embeddings**

Default architecture: 6 layers, 6 heads, 64 embedding dim.

### Tokenizer (`src/tokenizer.py`)

Character-level tokenizer with encode/decode methods. Builds vocabulary from corpus.

## GPU Support

The training script automatically detects available hardware:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

For Apple Silicon (MPS), modify `src/lm/utils.py`:
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### Remote Training (Windows + AMD GPU)

SSH into your Windows WSL2 instance from Mac:

```bash
# Connect to WSL2 via Windows IP
ssh <wsl-username>@<WINDOWS_IP>

# Navigate and train
cd ~/tiny-trons
python src/lm/train.py

# Monitor GPU (in separate SSH session)
watch -n 1 rocm-smi
```

See `TODO.md` for full WSL2 + ROCm setup instructions.

## License

MIT
