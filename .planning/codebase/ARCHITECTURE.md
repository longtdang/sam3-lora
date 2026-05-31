<!-- refreshed: 2025-01-31 -->
# Architecture

**Analysis Date:** 2025-01-31

## Overview

SAM3_LoRA is a LoRA (Low-Rank Adaptation) fine-tuning framework layered on top of Meta's SAM3 (Segment Anything Model 3) architecture. The project enables parameter-efficient fine-tuning of SAM3 on custom segmentation datasets by injecting trainable low-rank matrices into frozen base model weights. The system follows a vision-language grounding approach: given an image and a text/box prompt, it produces instance segmentation masks.

The architecture has three distinct layers:
1. **SAM3 Base Model** (`sam3/`) — Meta's original model code, frozen during training
2. **LoRA Injection Layer** (`lora_layers.py`, `sam3_lora/lora/`, `src/lora/`) — adds trainable adapters
3. **Training/Inference Orchestration** (root-level scripts) — multiple competing implementations

---

## System Overview

```text
┌───────────────────────────────────────────────────────────────────────┐
│                  Entry Points (root-level scripts)                     │
│  train_sam3_lora_native.py  train.py  train_standalone.py  infer_sam.py │
└────────────────┬──────────────────────────────────┬───────────────────┘
                 │                                  │
                 ▼                                  ▼
┌──────────────────────────┐          ┌─────────────────────────────────┐
│   LoRA Injection Layer   │          │   SAM3 Model Builder            │
│   `lora_layers.py`       │◀────────▶│   `sam3/model_builder.py`       │
│   `sam3_lora/lora/`      │          │   `sam3/__init__.py`            │
│   `src/lora/`            │          └────────────────┬────────────────┘
└──────────────────────────┘                           │
                                                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│                         SAM3 Model (`sam3/model/`)                     │
│                                                                        │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐  │
│  │  SAM3VLBackbone      │    │  Sam3Image (main model)              │  │
│  │  `vl_combiner.py`   │    │  `sam3_image.py`                     │  │
│  │                     │    │                                       │  │
│  │  ┌───────────────┐  │    │  ┌────────────────────────────────┐  │  │
│  │  │ViT backbone   │  │    │  │ TransformerEncoderFusion (DETR) │  │  │
│  │  │`vitdet.py`    │  │    │  │ `encoder.py`  (6 layers)       │  │  │
│  │  │32-layer, d=1024│ │    │  └────────────────────────────────┘  │  │
│  │  └───────────────┘  │    │  ┌────────────────────────────────┐  │  │
│  │  ┌───────────────┐  │    │  │ TransformerDecoder (DETR)      │  │  │
│  │  │VIT FPN Neck   │  │    │  │ `decoder.py`  (6 layers)       │  │  │
│  │  │`necks.py`     │  │    │  │  200 object queries            │  │  │
│  │  └───────────────┘  │    │  └────────────────────────────────┘  │  │
│  │  ┌───────────────┐  │    │  ┌────────────────────────────────┐  │  │
│  │  │VETextEncoder  │  │    │  │ SeqGeometryEncoder             │  │  │
│  │  │`text_encoder_ve│ │    │  │ `geometry_encoders.py` (3 lay) │  │  │
│  │  │ .py` (CLIP)   │  │    │  └────────────────────────────────┘  │  │
│  │  └───────────────┘  │    │  ┌────────────────────────────────┐  │  │
│  └─────────────────────┘    │  │ Mask Decoder / PixelDecoder    │  │  │
│                             │  │ `maskformer_segmentation.py`   │  │  │
│                             │  └────────────────────────────────┘  │  │
│                             └──────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   Training Infrastructure (`sam3/train/`)              │
│  data/ · loss/ · optim/ · transforms/ · utils/  +  trainer.py         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `Sam3Image` | Main model: orchestrates forward pass, auxiliary outputs | `sam3/model/sam3_image.py` |
| `SAM3VLBackbone` | Wraps vision + text backbone, handles activation checkpointing | `sam3/model/vl_combiner.py` |
| `ViT` | Vision Transformer backbone (32-layer, 1024-dim, 1008×1008 @ patch_size=14) | `sam3/model/vitdet.py` |
| `Sam3DualViTDetNeck` | FPN-style neck, produces multi-scale features from ViT | `sam3/model/necks.py` |
| `VETextEncoder` | CLIP-style text encoder for concept prompts | `sam3/model/text_encoder_ve.py` |
| `TransformerEncoderFusion` | DETR encoder (6 layers, d_model=256): fuses visual + text features | `sam3/model/encoder.py` |
| `TransformerDecoder` | DETR decoder (6 layers, 200 object queries): produces per-instance features | `sam3/model/decoder.py` |
| `SequenceGeometryEncoder` | Encodes box/point geometry prompts (3 layers) | `sam3/model/geometry_encoders.py` |
| `PixelDecoder` / `UniversalSegmentationHead` | Mask decoder: 3-stage upsampling from object queries to pixel masks | `sam3/model/maskformer_segmentation.py` |
| `build_sam3_image_model` | Factory that assembles all components into `Sam3Image` | `sam3/model_builder.py` |
| `LoRAConfig` | Configuration dataclass for LoRA rank, alpha, target modules, component flags | `lora_layers.py` (root), `sam3_lora/lora/lora_utils.py`, `src/lora/lora_utils.py` |
| `apply_lora_to_model` | Injects LoRA by (1) replacing `nn.MultiheadAttention` → `MultiheadAttentionLoRA`, (2) wrapping `nn.Linear` → `LoRALinear` | `lora_layers.py` |
| `MultiheadAttentionLoRA` | Drop-in MHA replacement with separate Q/K/V projections so LoRA can be applied | `lora_layers.py` |
| `LoRALayer` / `LoRALinear` | Low-rank adapter: W' = W + (B @ A) × (alpha/rank); base weights frozen | `lora_layers.py`, `sam3_lora/lora/lora_layer.py` |
| `Sam3LossWrapper` | Loss orchestration: bbox (L1+GIoU), cls (focal BCE), mask (focal+dice) | `sam3/train/loss/sam3_loss.py` |
| `BinaryHungarianMatcherV2` | Hungarian matching for one-to-one target assignment | `sam3/train/matcher.py` |
| `Trainer` | Official SAM3 full trainer with Hydra config, DDP, checkpointing | `sam3/train/trainer.py` |
| `LoRATrainer` | LoRA-specific trainer over SAM3 model, TensorBoard logging, AMP | `src/train/train_lora.py` |
| `Sam3ImageDataset` | Loads COCO-format annotations with optional segmentation masks | `sam3/train/data/sam3_image_dataset.py` |

---

## Design Patterns

### 1. LoRA Adapter Pattern (Wrapper/Decorator)
The LoRA injection is non-invasive: `apply_lora_to_model()` modifies the existing `Sam3Image` model graph in-place by replacing specific `nn.Module` instances. The original weights are preserved and frozen; only the two low-rank matrices `lora_A` (r × in) and `lora_B` (out × r) are trainable.

```python
# lora_layers.py — LoRALinear forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.original_layer(x) + self.lora(x)  # W·x + (B·A)·x·scale

# lora_layers.py — LoRALayer forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
    return lora_out * self.scaling  # scaling = alpha / rank
```

### 2. Two-Step MHA Replacement
Because `nn.MultiheadAttention` fuses Q/K/V into a single `in_proj_weight`, LoRA cannot be applied independently to each projection. `apply_lora_to_model()` first replaces all `nn.MultiheadAttention` with `MultiheadAttentionLoRA` (which has separate `q_proj`, `k_proj`, `v_proj`, `out_proj`), then applies `LoRALinear` to each:

```python
# lora_layers.py:413-446 — Step 1: Replace MHA
new_mha = MultiheadAttentionLoRA(embed_dim=..., num_heads=..., in_proj_weight=mha.in_proj_weight, ...)
setattr(parent, attr_name, new_mha)

# lora_layers.py:452-468 — Step 2: Wrap Linear → LoRALinear
lora_linear = LoRALinear(module, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
setattr(parent, attr_name, lora_linear)
```

### 3. Component-Level LoRA Gating
`LoRAConfig` carries boolean flags (`apply_to_vision_encoder`, `apply_to_text_encoder`, `apply_to_detr_encoder`, `apply_to_detr_decoder`, `apply_to_geometry_encoder`, `apply_to_mask_decoder`) to selectively target specific SAM3 sub-architectures. This allows lightweight "attention-only" or "DETR-only" fine-tuning modes.

### 4. Hydra-Driven Configuration (Official Path)
The official SAM3 trainer (`sam3/train/trainer.py`) uses Hydra + OmegaConf for config composition. The LoRA configs in `sam3_lora_configs/*.yaml` extend this system using `_target_: sam3_lora_wrapper.wrap_sam3_model_with_lora` to inject LoRA at model construction time.

### 5. Factory Pattern for Model Assembly
`build_sam3_image_model()` in `sam3/model_builder.py` is the single factory function assembling all SAM3 sub-components. All training scripts call this function before applying LoRA.

### 6. Auxiliary Output Pattern
`Sam3Image.forward()` returns both final predictions and intermediate decoder layer outputs (`aux_outputs`). Loss is computed across all decoder layers using `_update_out()` helper. This is standard DETR-style training with auxiliary losses.

---

## Data Flow

### Training Data Flow

```
COCO JSON annotation file
        │
        ▼
Sam3ImageDataset (`sam3/train/data/sam3_image_dataset.py`)
  - loads images + bboxes + optional RLE masks
  - applies transforms pipeline (resize to 1008, decode RLE, normalize)
        │
        ▼
collate_fn_api (`sam3/train/data/collator.py`)
  - batches Datapoint objects → BatchedDatapoint
        │
        ▼
Sam3Image.forward(samples, targets)
  │
  ├─► SAM3VLBackbone
  │      ├─► ViT (32-layer) → feature map (B, C, H/14, W/14)
  │      ├─► Sam3DualViTDetNeck → multi-scale FPN features
  │      └─► VETextEncoder → text embeddings (B, seq_len, 256)
  │
  ├─► SequenceGeometryEncoder → geometry prompt embeddings
  │
  ├─► TransformerEncoderFusion (6 layers)
  │      - self-attention on visual features
  │      - cross-attention with text embeddings
  │      → enhanced image features
  │
  ├─► TransformerDecoder (6 layers, 200 object queries)
  │      - cross-attention: queries attend to image features
  │      - cross-attention: queries attend to text features
  │      → per-instance query embeddings (B, 200, 256)
  │      → aux_outputs for each decoder layer
  │
  └─► MaskPredictor / UniversalSegmentationHead
         → instance masks (B, N, H, W)
         → bounding boxes (B, N, 4) in cx/cy/w/h
         → class logits (B, N, 1)
        │
        ▼
Sam3LossWrapper
  - Hungarian matching (BinaryHungarianMatcherV2)
  - bbox loss: L1 + GIoU (weight 5.0, 2.0)
  - class loss: focal BCE (weight 20.0)
  - mask loss: focal + dice (weight 200.0, 10.0)
  - computed for final + all aux outputs
        │
        ▼
AdamW optimizer — only LoRA parameters have requires_grad=True
```

### Inference Data Flow

```
Image + Text/Box prompt
        │
        ▼
build_sam3_image_model() → Sam3Image (frozen)
apply_lora_to_model(model, LoRAConfig) → Sam3Image + LoRA adapters
load_lora_weights(model, weights_path) → loads .pt file (strict=False)
        │
        ▼
Transform: resize to 1008, normalize [0.5, 0.5, 0.5]
collate_fn_api → BatchedDatapoint
        │
        ▼
model.forward() → raw predictions
        │
        ▼
PostProcessImage (`sam3/eval/postprocessors.py`)
  - converts cx/cy/w/h → x1/y1/x2/y2
  - applies score thresholds
  - rescales to original image size
        │
        ▼
Rendered masks / bounding boxes
```

---

## System Boundaries

### External Dependencies
- **PyTorch** — core ML framework, `torch.nn.Module` hierarchy
- **HuggingFace Hub** — model weights downloaded via `hf_hub_download()` in `model_builder.py`
- **Hydra / OmegaConf** — config management in the official trainer (`sam3/train/trainer.py`)
- **iopath** — file I/O abstraction (Meta internal) used in trainer and model builder
- **timm** — `DropPath`, `Mlp`, `trunc_normal_` in `vitdet.py`
- **torchvision** — `RoIAlign`, `nms`, `v2` transforms

### Model Weight Sources
- Base SAM3 weights: loaded from HuggingFace Hub (`facebook/sam3`, `load_from_HF=True`)
- LoRA delta weights: saved/loaded as lightweight `.pt` state dicts (only `lora_A` + `lora_B` tensors)

### LoRA Layer Targeting
LoRA is applied to these layer name patterns (configurable via `LoRAConfig.target_modules`):
- Vision backbone: `qkv`, `proj`, `fc1`, `fc2` (ViT attention + FFN)
- Text backbone: `c_fc`, `c_proj` (CLIP MLP)
- DETR encoder/decoder: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `linear1`, `linear2`

### Trainable vs Frozen
- **Frozen**: All original `sam3/` model parameters (base weights)
- **Trainable**: Only `lora_A` and `lora_B` matrices in injected `LoRALayer`/`LinearWithLoRA` modules

---

## Error Handling

**Strategy:** Scripts use try/except blocks at the training loop level; no custom exception hierarchy.

**Patterns:**
- Gradient clipping (`max_grad_norm=1.0`) guards against exploding gradients during LoRA updates
- `strict=False` in `model.load_state_dict()` allows partial LoRA weight loading without crashing on missing base weights
- AMP (Automatic Mixed Precision) with `GradScaler` in native training scripts (`train_sam3_lora_native.py`)

---

## Architectural Constraints

- **Threading:** PyTorch DataLoader multi-process (`num_workers` configurable, default 4–10). DDP training via `torch.distributed` with NCCL backend.
- **Global state:** `_setup_tf32()` sets `torch.backends.cuda.matmul.allow_tf32 = True` as a module-level side effect in `sam3/model_builder.py`.
- **Resolution:** Hard-coded input resolution of **1008×1008** throughout the stack (ViT `img_size=1008`, transform `PadToSizeAPI`).
- **Duplicate LoRA implementations:** Three parallel LoRA implementations exist (`lora_layers.py`, `sam3_lora/lora/`, `src/lora/`) with slightly different APIs — see CONCERNS.md.

---

## Anti-Patterns

### Parallel LoRA Implementations
**What happens:** `LoRAConfig`, `LoRALayer`, and `inject_lora_into_model` are duplicated across `lora_layers.py` (root), `sam3_lora/lora/lora_utils.py`, and `src/lora/lora_utils.py`. Different training scripts import from different locations.
**Why it's wrong:** Divergent behavior and parameter incompatibilities; `lora_layers.py` has `MultiheadAttentionLoRA` replacement logic absent in the others.
**Do this instead:** Consolidate into `sam3_lora/lora/` and update all imports.

### Multiple Competing Training Entry Points
**What happens:** Seven root-level training scripts exist (`train.py`, `train_simple.py`, `train_standalone.py`, `train_native.py`, `train_sam3_lora.py`, `train_sam3_lora_native.py`, `train_sam3_lora_with_categories.py`).
**Why it's wrong:** Unclear canonical entry point; maintainability burden.
**Do this instead:** `train_sam3_lora_native.py` is the most complete native implementation; `sam3_lora_configs/lora_base.yaml` + `train_native.py` is the Hydra path.

---

*Architecture analysis: 2025-01-31*
