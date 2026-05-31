# Code Conventions

**Analysis Date:** 2025-07-14

## Naming Conventions

**Files:**
- Snake_case for all Python files: `lora_layer.py`, `lora_utils.py`, `train_utils.py`
- Training scripts prefixed with `train_`: `train_sam3_lora.py`, `train_sam3_lora_native.py`, `train_simple.py`
- Inference scripts prefixed with `infer_` or `inference_`: `infer_sam.py`, `inference_lora.py`
- Test scripts prefixed with `test_`: `test_lora_injection.py`, `test_aux_outputs.py`
- Utility scripts named descriptively: `prepare_data.py`, `validate_sam3_lora.py`, `compare_lora_base.py`

**Classes:**
- PascalCase throughout: `LoRAConfig`, `LoRALayer`, `LoRALinear`, `LinearWithLoRA`, `MultiheadAttentionLoRA`, `SimpleLoRATrainer`, `TrainingConfig`, `COCOSegmentDataset`
- Config classes suffixed with `Config`: `LoRAConfig`, `TrainingConfig`
- Trainer classes suffixed with `Trainer`: `SimpleLoRATrainer`
- Dataset classes suffixed with `Dataset`: `COCOSegmentDataset`, `SimpleSAM3Dataset`

**Functions:**
- Snake_case: `inject_lora_into_model()`, `get_lora_parameters()`, `print_trainable_parameters()`, `count_parameters()`
- Private/internal helpers prefixed with `_`: `_should_inject_lora()`, `_is_inside_multihead_attention()`, `_sa_block()`, `_ff_block()`
- Action verbs for mutating functions: `apply_lora_to_model()`, `save_lora_weights()`, `load_lora_weights()`, `merge_lora_weights()`

**Variables:**
- Snake_case: `lora_config`, `lora_params`, `train_loader`, `batch_size`
- Abbreviations for tensors are common: `q`, `k`, `v`, `attn_weights`, `src`, `tgt`
- Model components follow PyTorch naming: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `lora_A`, `lora_B`

**Module attributes / LoRA weights:**
- LoRA matrices use uppercase single letters: `lora_A`, `lora_B` (follows the paper notation)
- Scaling factor named `scaling = alpha / rank`

## Code Style

**Line Length:**
- No explicit configuration found (no `.flake8` or `pyproject.toml`)
- Observed maximum ~120 chars in practice; long lines appear in config dictionary access and inline comments

**Indentation:**
- 4 spaces throughout (standard Python)

**Blank Lines:**
- Two blank lines between top-level definitions (classes, functions)
- One blank line between methods within a class
- Logical groups within methods separated by blank lines with inline section comments

**Trailing Commas:**
- Used consistently in multi-line function signatures:
  ```python
  encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=1024,
      batch_first=True,
  )
  ```

**String Formatting:**
- f-strings used universally for runtime messages:
  ```python
  print(f"Replaced {len(mha_replaced)} nn.MultiheadAttention modules with MultiheadAttentionLoRA")
  logging.info(f"Resuming training from {ckpt_path}")
  ```

**Visual Separators:**
- Section headers in scripts use `print("=" * 60)` and `print("-" * 60)` 
- Unicode checkmarks in output: `✓ Forward pass successful!`, `✗ Forward pass failed:`

## Patterns & Idioms

**PyTorch Module Pattern:**
All neural-network classes extend `nn.Module` and define `__init__` + `forward`:
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, ...):
        super().__init__()
        # parameter registration
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

**Configuration Objects:**
Config state is held in plain classes (not dataclasses), with `__init__` that sets all attributes directly from args:
```python
class LoRAConfig:
    def __init__(self, rank: int = 4, alpha: float = 1.0, ...):
        self.rank = rank
        self.alpha = alpha
        ...
    def to_dict(self) -> Dict:
        return {"rank": self.rank, "alpha": self.alpha, ...}
```
Config objects also include a `to_dict()` method for serialization.

**Dataclasses:**
Used in core SAM3 data structures (`sam3/train/trainer.py`, `sam3/train/data/sam3_image_dataset.py`) via `@dataclass` and `@field`.

**Freezing Parameters:**
Standard LoRA pattern — freeze base model first, then inject trainable adapters:
```python
for param in model.parameters():
    param.requires_grad = False
# ... then wrap linear layers with LoRALinear which adds trainable lora_A, lora_B
```

**Module Replacement:**
LoRA injection uses the `setattr(parent, attr_name, new_module)` pattern to swap out `nn.Linear` and `nn.MultiheadAttention` in-place:
```python
*parent_path, attr_name = name.split('.')
parent = model
for p in parent_path:
    parent = getattr(parent, p)
setattr(parent, attr_name, lora_linear)
```

**Context Managers:**
`torch.no_grad()` used during evaluation and test forward passes:
```python
with torch.no_grad():
    output = model(src, tgt)
```

**Script Entry Point:**
All executable scripts use `if __name__ == "__main__":` guard, typically calling a `main()` function or invoking `argparse`:
```python
if __name__ == "__main__":
    main()
```

**Error Handling:**
Try/except used in test scripts to catch forward/backward pass failures, with `traceback.print_exc()` for debugging:
```python
try:
    output = model(src, tgt)
    print(f"✓ Forward pass successful!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    return
```

**Copyright Header:**
Meta Platforms code (`sam3/`) carries license header:
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
```
Project-added files (`src/`, `sam3_lora/`, root scripts) use module-level docstrings without a copyright header.

## Documentation Standards

**Module Docstrings:**
All Python files begin with a triple-quoted module docstring describing purpose:
```python
"""
LoRA (Low-Rank Adaptation) Layer Implementation for SAM3

This module implements LoRA layers that can be injected into transformer models
for efficient fine-tuning.
"""
```

**Class Docstrings:**
Classes include a summary line and an `Args:` section listing constructor parameters:
```python
class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear transformation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of LoRA matrices (r in the paper)
        alpha: Scaling factor (typically set to rank)
        dropout: Dropout probability for LoRA weights
    """
```

**Method Docstrings:**
`forward()` methods have a brief one-line or short paragraph docstring:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply LoRA transformation: x @ (A @ B) * scaling
    """
```

**Inline Comments:**
Heavy use of inline `#` comments for section labels within long functions:
```python
# STEP 1: Replace nn.MultiheadAttention with MultiheadAttentionLoRA
# STEP 2: Apply LoRA to all matching Linear layers
# CRITICAL: Freeze all base model parameters first
```
`CRITICAL:` and `NOTE:` prefixes mark important constraints.

**Type Hints:**
Consistently used on all public functions and methods. Both parameter types and return types annotated:
```python
def inject_lora_into_model(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = False,
) -> nn.Module:
```
`Optional`, `List`, `Dict`, `Set`, `Tuple` imported from `typing`.

## Linting & Formatting

**Formatters Declared (not enforced via CI):**
- `black>=23.3.0` listed in `requirements.txt` and `setup.py` dev extras
- No `.black` config file detected; default 88-char line length assumed

**Linters Declared (not enforced via CI):**
- `flake8>=6.0.0` listed in `requirements.txt` and `setup.py` dev extras
- No `.flake8` config file detected; default rules apply

**Not Present:**
- `mypy` — no static type checking configured
- `isort` — no import sort config
- `pylint` — not in requirements
- `pyproject.toml` — absent
- `pre-commit` hooks — absent
- CI pipeline — no `.github/workflows/` directory found

**Import Ordering (observed pattern):**
1. Standard library (`os`, `math`, `re`, `json`, `time`, `logging`, `contextlib`)
2. Third-party (`numpy`, `torch`, `torch.nn`, `PIL`, `tqdm`, `yaml`, `hydra`)
3. Internal project (`from .lora_layer import ...`, `from sam3.model...`, `from sam3_lora.lora...`)
Blank line separates each group in well-organized files; some scripts mix groups without separation.

---

*Convention analysis: 2025-07-14*
