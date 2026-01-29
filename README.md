# SteeringLLM

Runtime LLM behavior modification through activation steering.

## Overview

SteeringLLM enables you to modify LLM behavior at inference time without retraining. Apply steering vectors to guide model outputs toward desired characteristics (e.g., more helpful, safer, more creative).

## Quick Start

```python
from steering_llm import SteeringModel, Discovery

# Load model
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# Create steering vector from contrast examples
vector = Discovery.mean_difference(
    positive=["I love helping people!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model,
    layer=15
)

# Generate with steering
output = model.generate_with_steering(
    "Tell me about yourself",
    vector=vector,
    alpha=2.0,
    max_length=100
)
```

## Installation

```bash
pip install steering-llm
```

### Development Installation

```bash
git clone https://github.com/jnPiyush/SteeringLLM.git
cd SteeringLLM
pip install -e ".[dev]"
```

## Features

- ✅ **Story #4**: Create steering vectors from contrast datasets
- 🚧 **Story #5**: Apply steering vectors at inference time
- 🚧 **Story #6**: Remove steering vectors

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.36+

## Architecture

See [docs/adr/ADR-001-steeringllm-architecture.md](docs/adr/ADR-001-steeringllm-architecture.md) for architectural decisions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see LICENSE file for details.
