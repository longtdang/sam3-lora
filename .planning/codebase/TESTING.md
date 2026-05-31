# Testing

**Analysis Date:** 2025-07-14

## Test Framework

**Runner:**
- `pytest>=7.4.0` declared in `requirements.txt` and `setup.py` dev extras
- No `pytest.ini`, `pyproject.toml`, or `tox.ini` configuration files present
- Default pytest discovery rules apply (files matching `test_*.py` or `*_test.py`)

**Assertion Library:**
- `pytest` native assertions (`assert`)
- `torch.testing.assert_close()` for tensor equality checks
- Standard Python `assert` in unit test classes

**Run Commands:**
```bash
# Run all tests (pytest-discovered)
pytest

# Run a specific test file
pytest sam3/perflib/tests/tests.py

# Run root-level standalone test scripts directly
python test_lora_injection.py
python test_aux_outputs.py

# Run with verbose output
pytest -v

# Run and show print output
pytest -s
```

No `Makefile`, `tox.ini`, or CI workflow is present to provide canonical test commands beyond the above.

## Test Structure

**Location:**
Three separate test contexts exist in this repository:

1. **pytest-compatible unit tests** — `sam3/perflib/tests/tests.py`
   - Uses `pytest` class-based style with `class TestX:` and `def test_*:` methods
   - Has a companion `sam3/perflib/tests/assets/` directory for test fixtures (TIFF mask files)

2. **Standalone integration test scripts** (root level) — run directly as Python scripts, not via pytest:
   - `test_lora_injection.py` — verifies LoRA injection forward/backward passes on a synthetic model
   - `test_aux_outputs.py` — checks auxiliary decoder outputs in train vs eval mode using real data

3. **Inline validation inside `sam3/perflib/tests/tests.py`** — only the perflib tests are pytest-native

**File Naming:**
- pytest tests: `tests.py` inside a `tests/` subdirectory
- Standalone test scripts: `test_<feature>.py` at repo root (e.g., `test_lora_injection.py`, `test_aux_outputs.py`)

**Directory Layout:**
```
sam3/perflib/tests/
├── assets/
│   └── masks.tiff        # Ground truth mask fixture
└── tests.py              # pytest class-based unit tests

test_lora_injection.py    # Root: standalone functional test
test_aux_outputs.py       # Root: standalone integration test
```

## Test Patterns

**pytest Class-Based (sam3/perflib/tests/tests.py):**
```python
class TestMasksToBoxes:
    def test_masks_box(self):
        def masks_box_check(masks, expected, atol=1e-4):
            out = masks_to_boxes(masks, [1 for _ in range(masks.shape[0])])
            assert out.dtype == torch.float
            torch.testing.assert_close(
                out, expected, rtol=0.0, check_dtype=True, atol=atol
            )
        # ... build expected tensor, call helper, assert
```
- Inner helper functions defined inside test methods for repetitive assertion logic
- No `setup`/`teardown` methods used

**Standalone Script Pattern (`test_lora_injection.py`):**
```python
def main():
    print("=" * 60)
    print("Testing LoRA Injection")
    print("=" * 60)

    # Numbered steps: 1. Create model, 2. Configure LoRA, 3. Inject, 4. Forward, 5. Backward
    try:
        output = model(src, tgt)
        print(f"✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
```
- Tests are organized as numbered steps printed to stdout
- `try/except` replaces assertions; `✓`/`✗` prefix indicates pass/fail
- No pytest assertions; results communicated via stdout

**Gradient Verification Pattern:**
Used in `test_lora_injection.py` to confirm LoRA parameters receive gradients and frozen parameters do not:
```python
loss.backward()
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            print(f"  Warning: Trainable param {name} has no gradient")
    else:
        if param.grad is not None:
            print(f"  Warning: Frozen param {name} has gradient!")
```

**Tensor Assertion Pattern (`sam3/perflib/tests/tests.py`):**
```python
torch.testing.assert_close(
    out, expected, rtol=0.0, check_dtype=True, atol=1e-4
)
```

**Multi-dtype Testing:**
Iteration over dtypes to ensure dtype-agnostic correctness:
```python
for dtype in [torch.float16, torch.float32, torch.float64]:
    masks = torch.zeros((image.n_frames, image.height, image.width), dtype=dtype)
    masks = _create_masks(image, masks)
    masks_box_check(masks, expected)
```

**Integration/Model State Testing (`test_aux_outputs.py`):**
Tests the same model in both `train()` and `eval()` modes to compare output shapes:
```python
trainer.model.train()
with torch.no_grad():
    outputs_list = trainer.model(input_batch)
# ... inspect outputs

trainer.model.eval()
with torch.no_grad():
    outputs_list = trainer.model(input_batch)
# ... compare
```

## Running Tests

**pytest (unit tests only):**
```bash
# Run all pytest-discovered tests
pytest

# Run only the perflib unit test
pytest sam3/perflib/tests/tests.py

# Verbose
pytest -v sam3/perflib/tests/tests.py

# With stdout
pytest -s sam3/perflib/tests/tests.py
```

**Standalone test scripts:**
```bash
# LoRA injection test (requires src.lora module)
python test_lora_injection.py

# Auxiliary outputs test (requires full SAM3 + data)
python test_aux_outputs.py
```

**Prerequisites for `test_aux_outputs.py`:**
- SAM3 model checkpoint available
- COCO-format data in `data/train/`
- `configs/full_lora_config.yaml` present

**Prerequisites for `test_lora_injection.py`:**
- No data required; uses synthetic random tensors
- Only needs `src.lora.lora_utils` to be importable

## Coverage

**Requirements:** None enforced — no coverage configuration file, no CI coverage gate.

**View Coverage (manual):**
```bash
# Install coverage if needed
pip install pytest-cov

# Run with coverage
pytest --cov=sam3_lora --cov=src sam3/perflib/tests/tests.py

# HTML report
pytest --cov=sam3_lora --cov=src --cov-report=html sam3/perflib/tests/tests.py
```

**Observed Coverage Gaps:**
- `src/lora/lora_layer.py` and `src/lora/lora_utils.py` — no pytest-discoverable tests (only manual script `test_lora_injection.py`)
- `sam3_lora/` package (trainer, lora layers, model utils) — no automated tests
- `train_sam3_lora_native.py`, `validate_sam3_lora.py` — no test coverage
- Only `sam3/perflib/masks_ops.py` has formal pytest coverage via `sam3/perflib/tests/tests.py`

---

*Testing analysis: 2025-07-14*
