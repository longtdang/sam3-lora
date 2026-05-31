# Project Structure

**Analysis Date:** 2025-01-31

## Directory Layout

```
SAM3_LoRA/                          # Project root
│
├── sam3/                           # SAM3 base model (Meta, Apache-licensed)
│   ├── __init__.py                 # Exports build_sam3_image_model
│   ├── model_builder.py            # Factory: assembles all SAM3 components
│   ├── logger.py                   # Logging utilities
│   ├── visualization_utils.py      # Visualization helpers
│   ├── assets/                     # BPE vocabulary file (bpe_simple_vocab_16e6.txt.gz)
│   ├── model/                      # All neural network modules
│   │   ├── sam3_image.py           # Sam3Image: main model class
│   │   ├── vl_combiner.py          # SAM3VLBackbone: vision+language combiner
│   │   ├── vitdet.py               # ViT backbone (ViTDet, 32-layer)
│   │   ├── necks.py                # Sam3DualViTDetNeck: FPN neck
│   │   ├── encoder.py              # TransformerEncoderFusion: DETR encoder (6L)
│   │   ├── decoder.py              # TransformerDecoder: DETR decoder (6L)
│   │   ├── geometry_encoders.py    # SequenceGeometryEncoder: bbox/point encoder
│   │   ├── maskformer_segmentation.py  # PixelDecoder, UniversalSegmentationHead
│   │   ├── text_encoder_ve.py      # VETextEncoder: CLIP-style text encoder
│   │   ├── tokenizer_ve.py         # SimpleTokenizer: BPE tokenizer
│   │   ├── position_encoding.py    # PositionEmbeddingSine
│   │   ├── model_misc.py           # MLP, DotProductScoring, SAM3Output, helpers
│   │   ├── box_ops.py              # Box format conversion utilities
│   │   ├── data_misc.py            # BatchedDatapoint, data structures
│   │   ├── memory.py               # CXBlock, SimpleFuser (for video mode)
│   │   ├── edt.py                  # EDT (for interactive mode)
│   │   ├── io_utils.py             # I/O utilities
│   │   ├── act_ckpt_utils.py       # Activation checkpointing wrappers
│   │   ├── sam3_image_processor.py # Image preprocessing
│   │   ├── sam3_tracker_base.py    # Video tracking base
│   │   ├── sam3_tracker_utils.py   # Tracking utilities
│   │   ├── sam3_tracking_predictor.py  # Tracking predictor
│   │   ├── sam3_video_base.py      # Video inference base
│   │   ├── sam3_video_inference.py # Video inference logic
│   │   ├── sam3_video_predictor.py # Video predictor class
│   │   ├── sam1_task_predictor.py  # SAM1-compatible interactive predictor
│   │   └── utils/
│   │       ├── misc.py             # copy_data_to_device and misc helpers
│   │       ├── sam1_utils.py       # SAM1 compatibility utilities
│   │       └── sam2_utils.py       # SAM2 compatibility utilities
│   ├── sam/                        # SAM1-style transformer components
│   │   ├── transformer.py          # RoPEAttention (used in SAM1 interactive mode)
│   │   ├── mask_decoder.py         # SAM1-style mask decoder
│   │   ├── prompt_encoder.py       # SAM1 prompt encoder
│   │   ├── rope.py                 # Rotary position embedding
│   │   └── common.py               # Shared SAM primitives
│   ├── train/                      # Official SAM3 training infrastructure
│   │   ├── trainer.py              # Trainer: Hydra/DDP trainer (canonical)
│   │   ├── train.py                # Train entry via Hydra launcher
│   │   ├── matcher.py              # BinaryHungarianMatcherV2, BinaryOneToManyMatcher
│   │   ├── masks_ops.py            # RLE encoding, mask operations
│   │   ├── nms_helper.py           # NMS utilities
│   │   ├── data/
│   │   │   ├── sam3_image_dataset.py   # Sam3ImageDataset (COCO format)
│   │   │   ├── sam3_video_dataset.py   # Video dataset
│   │   │   ├── torch_dataset.py        # TorchDataset wrapper
│   │   │   ├── coco_json_loaders.py    # COCO JSON parsing
│   │   │   └── collator.py             # collate_fn_api, BatchedDatapoint
│   │   ├── loss/
│   │   │   ├── sam3_loss.py            # Sam3LossWrapper, DummyLoss
│   │   │   ├── loss_fns.py             # IABCEMdetr, Boxes, Masks
│   │   │   ├── mask_sampling.py        # Mask sampling strategies
│   │   │   └── sigmoid_focal_loss.py   # Focal loss implementation
│   │   ├── optim/
│   │   │   ├── optimizer.py            # construct_optimizer, GradientClipper, layer_decay_param_modifier
│   │   │   └── schedulers.py           # InverseSquareRootParamScheduler
│   │   ├── transforms/
│   │   │   ├── basic.py                # get_random_resize_scales, helpers
│   │   │   ├── basic_for_api.py        # ComposeAPI, RandomResizeAPI, PadToSizeAPI, NormalizeAPI
│   │   │   ├── filter_query_transforms.py  # FlexibleFilterFindGetQueries
│   │   │   ├── point_sampling.py       # RandomizeInputBbox
│   │   │   └── segmentation.py         # DecodeRle
│   │   ├── utils/
│   │   │   ├── train_utils.py          # AverageMeter, set_seeds, setup_distributed_backend
│   │   │   ├── checkpoint_utils.py     # load_state_dict_into_model, checkpoint helpers
│   │   │   ├── distributed.py          # all_reduce_max, barrier, get_rank
│   │   │   └── logger.py               # Logger, setup_logging, make_tensorboard_logger
│   │   └── configs/                    # Eval config YAML files
│   │       ├── gold_image_evals/
│   │       ├── silver_image_evals/
│   │       ├── saco_video_evals/
│   │       ├── odinw13/
│   │       └── roboflow_v100/
│   ├── eval/                       # Evaluation toolkits
│   │   ├── coco_eval.py            # COCO AP evaluation
│   │   ├── coco_eval_offline.py    # Offline COCO evaluation
│   │   ├── coco_writer.py          # PredictionDumper
│   │   ├── postprocessors.py       # PostProcessImage (bbox rescaling, NMS)
│   │   ├── hota_eval_toolkit/      # HOTA tracking metrics
│   │   └── teta_eval_toolkit/      # TETA tracking metrics
│   ├── agent/                      # Interactive/agentic inference mode
│   │   ├── agent_core.py           # Core agent logic
│   │   ├── client_llm.py           # LLM client integration
│   │   ├── client_sam3.py          # SAM3 inference client
│   │   ├── inference.py            # Agent inference pipeline
│   │   ├── viz.py                  # Visualization
│   │   └── helpers/                # Boxes, masks, keypoints, zoom, memory helpers
│   └── perflib/                    # Performance library (Triton kernels, NMS)
│       ├── triton/                 # Triton GPU kernels
│       └── tests/                  # Performance tests
│
├── sam3_lora/                      # Standalone LoRA package (alternative implementation)
│   ├── __init__.py                 # Exports LoRAConfig, inject_lora_into_model
│   ├── lora/
│   │   ├── lora_layer.py           # LoRALayer, LinearWithLoRA
│   │   └── lora_utils.py           # LoRAConfig, inject_lora_into_model, get_lora_state_dict
│   ├── data/
│   │   └── dataset.py              # LoRASAM3Dataset (COCO format)
│   ├── model/
│   │   └── simple_models.py        # SimpleSegmentationModel (standalone demo)
│   ├── train/
│   │   ├── trainer.py              # SimpleLoRATrainer (generic, SAM3-free)
│   │   └── native_trainer.py       # Native LoRA trainer
│   └── utils/
│       └── training_utils.py       # print_trainable_parameters
│
├── src/                            # Alternate LoRA package (older implementation)
│   ├── __init__.py
│   ├── lora/
│   │   ├── lora_layer.py           # LoRALayer, LinearWithLoRA (mirrors sam3_lora/lora)
│   │   └── lora_utils.py           # LoRAConfig, inject_lora_into_model, get/load/save utils
│   ├── data/
│   │   └── dataset.py              # LoRASAM3Dataset (identical to sam3_lora/data)
│   └── train/
│       └── train_lora.py           # LoRATrainer (TensorBoard + SAM3 loss)
│
├── configs/                        # Config YAML for HuggingFace-style training
│   ├── base_config.yaml            # Conservative settings (rank=8, alpha=16)
│   ├── full_lora_config.yaml       # Full LoRA (all components enabled)
│   ├── light_lora_config.yaml      # Lightweight LoRA (fewer modules)
│   ├── minimal_lora_config.yaml    # Minimal LoRA (attention-only)
│   ├── crack_detection_config.yaml # Domain-specific config example
│   └── sam3_lora_standalone.yaml   # Standalone training config
│
├── sam3_lora_configs/              # Config YAML for Hydra/official trainer path
│   ├── lora_base.yaml              # Base Hydra config (extends sam3/train/trainer.py)
│   ├── lora_full.yaml              # Full LoRA Hydra config
│   └── lora_minimal.yaml           # Minimal LoRA Hydra config
│
│── lora_layers.py                  # PRIMARY LoRA implementation (root-level, canonical)
│                                   #   MultiheadAttentionLoRA, LoRALayer, LoRALinear,
│                                   #   LoRAConfig, apply_lora_to_model, save/load_lora_weights
│
├── train_sam3_lora_native.py       # Recommended: native PyTorch trainer, SAM3 loss, AMP
├── train_sam3_lora.py              # HuggingFace Sam3Model/Sam3Processor based trainer
├── train_sam3_lora_with_categories.py  # Category-aware native trainer
├── train.py                        # Trainer using src/ package
├── train_simple.py                 # Simplified trainer (educational)
├── train_standalone.py             # Trainer using sam3_lora/ package (SAM3-free demo)
├── train_native.py                 # Hydra-driven launcher (official SAM3 trainer + LoRA)
│
├── inference_lora.py               # Full inference: build_sam3_image_model + LoRA
├── inference.py                    # Basic inference wrapper
├── infer_sam.py                    # Extended inference with evaluation
├── compare_lora_base.py            # Single-image LoRA vs base comparison
├── compare_lora_base_batch.py      # Batch LoRA vs base comparison
├── validate_sam3_lora.py           # Validation script with COCO metrics
│
├── prepare_data.py                 # Data prep: Roboflow/COCO format conversion
├── prepare_data_split.py           # Train/val split utility
├── convert_roboflow_to_coco.py     # Roboflow → COCO JSON converter
│
├── analyze_loss.py                 # Training loss curve analysis
├── check_text_encoding.py          # Text encoder diagnostic
├── test_lora_injection.py          # LoRA injection unit test
├── test_aux_outputs.py             # Auxiliary output test
├── verify_gt_transforms.py         # Ground truth transform verification
│
├── setup.py                        # Package install (installs sam3 as editable)
├── requirements.txt                # Python dependencies
├── loral.yaml                      # Minimal paths config for experiments
│
├── asset/                          # Static assets (images for README/docs)
├── .planning/                      # GSD planning documents
│   └── codebase/                   # Codebase map documents (ARCHITECTURE.md, STRUCTURE.md)
│
├── README.md                       # Project overview
├── README_INFERENCE.md             # Inference guide
├── README_LORA_IMPLEMENTATION.md   # LoRA implementation deep-dive
├── LORA_IMPLEMENTATION_GUIDE.md    # Step-by-step guide
├── CLI_TRAINING_GUIDE.md           # CLI usage guide
├── PROJECT_SUMMARY.md              # Project summary
├── QUICK_SUMMARY.md                # Quick reference
└── diagnose_training.md            # Troubleshooting guide
```

---

## Key Files

### Entry Points

| File | Purpose |
|------|---------|
| `train_sam3_lora_native.py` | Most complete training script; native PyTorch, SAM3 loss stack, AMP, gradient clipping |
| `train_native.py` | Hydra-based launcher using official `sam3.train.trainer.Trainer` + LoRA wrap |
| `inference_lora.py` | Primary inference script: loads model, applies LoRA, runs forward pass |
| `infer_sam.py` | Extended inference with evaluation postprocessing |
| `sam3/model_builder.py` | Single factory for assembling `Sam3Image` — called by all training/inference scripts |

### Core LoRA Files

| File | Purpose |
|------|---------|
| `lora_layers.py` | **Canonical LoRA implementation** — `MultiheadAttentionLoRA`, `LoRALinear`, `LoRAConfig`, `apply_lora_to_model`, `save_lora_weights`, `load_lora_weights` |
| `sam3_lora/lora/lora_utils.py` | Alternative LoRA utilities (simpler, without MHA replacement) |
| `src/lora/lora_utils.py` | Older alternative (used by `train.py`, `train_simple.py`) |

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/base_config.yaml` | HuggingFace-style training config (rank=8, HF model name, data paths) |
| `sam3_lora_configs/lora_base.yaml` | Hydra training config (Trainer target, data loaders, loss, optimizer) |
| `loral.yaml` | Minimal paths override for quick experiments |

### SAM3 Model Core

| File | Purpose |
|------|---------|
| `sam3/model/sam3_image.py` | `Sam3Image` — the top-level model class; orchestrates forward pass |
| `sam3/model/vl_combiner.py` | `SAM3VLBackbone` — wraps ViT + text encoder |
| `sam3/model/vitdet.py` | `ViT` — 32-layer ViT backbone with RoPE |
| `sam3/model/encoder.py` | `TransformerEncoderFusion` — DETR encoder, 6 layers |
| `sam3/model/decoder.py` | `TransformerDecoder` — DETR decoder, 6 layers, 200 queries |
| `sam3/model/maskformer_segmentation.py` | `PixelDecoder`, `UniversalSegmentationHead` — mask output |
| `sam3/train/data/sam3_image_dataset.py` | `Sam3ImageDataset` — COCO format loader |
| `sam3/train/loss/sam3_loss.py` | `Sam3LossWrapper` — composite loss (bbox + cls + mask) |

---

## Module Organization

### Package Structure Summary

| Package | Role | Used By |
|---------|------|---------|
| `sam3` | Base SAM3 model (Meta code, do not edit) | All training/inference scripts |
| `lora_layers.py` (root) | Canonical LoRA implementation | `train_sam3_lora*.py`, `inference*.py`, `infer_sam.py`, `compare_lora*.py` |
| `sam3_lora` | Standalone LoRA package (no SAM3 dep) | `train_standalone.py` |
| `src` | Older LoRA + dataset package | `train.py`, `train_simple.py`, `test_lora_injection.py` |
| `configs/` | YAML configs for HuggingFace-style training | `train_sam3_lora.py` |
| `sam3_lora_configs/` | YAML configs for Hydra/official training | `train_native.py` |

### Import Paths by Script

```python
# Canonical native training (train_sam3_lora_native.py)
from sam3.model_builder import build_sam3_image_model
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks
from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights

# Hydra-based training (train_native.py / sam3_lora_configs/*.yaml)
from sam3.train.trainer import Trainer
# model wrapped via: sam3_lora_wrapper.wrap_sam3_model_with_lora

# src/ based training (train.py)
from src.lora.lora_utils import LoRAConfig
from src.data.dataset import create_dataloaders
from src.train.train_lora import LoRATrainer

# Standalone training (train_standalone.py)
from sam3_lora import LoRAConfig, inject_lora_into_model
from sam3_lora.model import SimpleSegmentationModel
```

---

## Naming Conventions

### Files
- **snake_case** for all Python files: `lora_layer.py`, `train_sam3_lora_native.py`
- **Module files** follow the convention `<concept>_<type>.py`: `sam3_image.py`, `train_utils.py`
- **Training scripts** prefixed `train_`: `train.py`, `train_sam3_lora.py`, `train_native.py`
- **Inference scripts** prefixed `infer_`/`inference_`: `infer_sam.py`, `inference_lora.py`
- **Config files**: `<scope>_config.yaml` for `configs/`, plain names for `sam3_lora_configs/`
- **Test/diagnostic scripts** prefixed `test_`/`check_`/`verify_`/`validate_`

### Classes
- **PascalCase** for all classes: `Sam3Image`, `LoRAConfig`, `SimpleLoRATrainer`
- **SAM3** prefix for main model classes: `Sam3Image`, `SAM3VLBackbone`
- **LoRA** prefix for adapter classes: `LoRAConfig`, `LoRALayer`, `LoRALinear`, `LoRATrainer`
- **Transformer** prefix for encoder/decoder: `TransformerEncoderFusion`, `TransformerDecoder`

### Config Keys
- **snake_case** YAML keys: `apply_to_vision_encoder`, `lr_scale`, `max_epochs`
- Hydra `_target_` fields use full dotted module paths: `sam3.train.trainer.Trainer`
- Hydra interpolation: `${scratch.resolution}`, `${training.batch_size}`

---

## Where to Add New Code

**New LoRA experiment / config variant:**
- Add YAML to `sam3_lora_configs/` (Hydra path) or `configs/` (HuggingFace path)
- Pattern: copy `sam3_lora_configs/lora_base.yaml`, adjust `lora.*` and `training.*` sections

**New LoRA layer type (e.g., LoRA for Conv2d):**
- Implement in `lora_layers.py` (root), adding a new class alongside `LoRALinear`
- Extend `apply_lora_to_model()` to handle `isinstance(module, nn.Conv2d)` case

**New training loss function:**
- Add to `sam3/train/loss/loss_fns.py` following the pattern of `Boxes`, `Masks`, `IABCEMdetr`
- Register in `sam3_lora_configs/lora_base.yaml` under `loss.loss_fns_find`

**New data transform:**
- Add to `sam3/train/transforms/basic_for_api.py` as a new `*API` class
- Register in config `train_transforms` list with `_target_: sam3.train.transforms.basic_for_api.YourTransform`

**New dataset format:**
- Implement a `Dataset` subclass in `sam3_lora/data/dataset.py` following `LoRASAM3Dataset`
- Use COCO format (JSON with `images`, `annotations`, `categories` keys)

**New inference script:**
- Start from `inference_lora.py` pattern: build model → apply LoRA → load weights → transform → forward → postprocess
- Use `sam3.model_builder.build_sam3_image_model()` for model construction
- Use `lora_layers.apply_lora_to_model()` + `lora_layers.load_lora_weights()` for LoRA

**New evaluation metric:**
- Add evaluator class to `sam3/eval/`
- Register in Hydra config under `trainer.meters.val.*`

---

## Special Directories

**`sam3/`:**
- Purpose: Original Meta SAM3 model code — treat as a frozen third-party library
- Generated: No
- Committed: Yes (vendored source)
- Do NOT edit unless fixing bugs; LoRA adapts it non-invasively

**`sam3/assets/`:**
- Purpose: BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) required for text tokenization
- Generated: No
- Must be present for inference/training; path configured via `bpe_path`

**`sam3/train/configs/`:**
- Purpose: Official SAM3 training eval configs (gold/silver/roboflow benchmarks)
- Generated: No
- Reference-only for standard SAM3 benchmarking

**`sam3/eval/hota_eval_toolkit/` and `sam3/eval/teta_eval_toolkit/`:**
- Purpose: Vendored tracking evaluation toolkits (HOTA, TETA metrics)
- Generated: No
- Used for video tracking evaluation mode only

**`.planning/codebase/`:**
- Purpose: GSD codebase map documents
- Generated: Yes (by GSD mapping agents)
- Committed: Yes

---

*Structure analysis: 2025-01-31*
