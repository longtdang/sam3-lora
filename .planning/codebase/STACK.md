# Technology Stack

**Analysis Date:** 2025-07-14

---

## Languages

**Primary:**
- Python 3.8+ (runtime: 3.12.3 on dev machine) — all training, inference, data processing, and model logic

**Secondary:**
- YAML — configuration files (`configs/`, `sam3_lora_configs/`, `sam3/train/configs/`)
- Bash — launch scripts (`quick_start.sh`, `quickstart.sh`, `run_5_image_comparison.sh`)

---

## Runtime & Build Tools

**Environment:**
- CPython 3.12.3 (minimum: 3.8)
- CUDA-capable GPU strongly recommended (16GB+ VRAM for LoRA training, 80GB+ for full fine-tune)
- NCCL backend for multi-GPU distributed training via `torch.distributed`

**Package Management:**
- pip (no lockfile; `requirements.txt` specifies minimum versions only)
- `setup.py` via `setuptools` / `find_packages()` — package name `sam3-lora`, version `0.1.0`
- No `pyproject.toml`, no `poetry.lock`, no `conda` environment file detected

**Build:**
- `setup.py` (`setuptools`) — installs the `sam3_lora` package and its dependencies
- No compiled extensions beyond what third-party packages provide
- `torch.compile` support noted in configs (`use_compile: false` by default)

---

## Frameworks & Libraries

### Deep Learning Core
| Package | Min Version | Role |
|---|---|---|
| `torch` | ≥2.7.0 | Core ML framework; used everywhere |
| `torchvision` | ≥0.19.0 | Image transforms (`torchvision.transforms.v2`), `nms`, dataset utils |
| `transformers` | ≥4.48.0 | `Sam3Model`, `Sam3Processor`, `get_scheduler` from HuggingFace |
| `einops` | ≥0.6.0 | Tensor reshaping in model code |
| `torchmetrics` | ≥1.0.0 | Evaluation metrics during training |

### LoRA / Parameter-Efficient Fine-Tuning
- Custom LoRA implementation in `lora_layers.py` and `sam3_lora/lora/`
- `LoRALayer`, `LinearWithLoRA`, `LoRAConfig`, `MultiheadAttentionLoRA`
- No external PEFT library (e.g., not using `peft` from HuggingFace)
- LoRA applied to `q_proj`, `k_proj`, `v_proj`, `out_proj` in attention layers

### Vision & Image Processing
| Package | Min Version | Role |
|---|---|---|
| `Pillow` | ≥10.0.0 | Image loading/saving |
| `opencv-python` | ≥4.8.0 | Image preprocessing, augmentation |
| `scikit-image` | ≥0.21.0 | Image processing utilities |
| `numpy` | ≥1.24.0 | Array operations throughout |
| `scipy` | ≥1.10.0 | Scientific computations |

### Configuration Management
| Package | Min Version | Role |
|---|---|---|
| `hydra-core` | ≥1.3.0 | Config composition for SAM3 native training (`sam3/train/`) |
| `omegaconf` | ≥2.3.0 | Structured config objects used with Hydra |
| `PyYAML` | ≥6.0 | Direct YAML config loading in `train_sam3_lora.py` scripts |

### Data & Evaluation
| Package | Min Version | Role |
|---|---|---|
| `pycocotools` | ≥2.0.6 | COCO annotation format parsing and evaluation |
| `pandas` | ≥1.5.0 | Data analysis and tabular processing |
| `scikit-learn` | ≥1.3.0 | Metrics, data splitting |

### Visualization & Logging
| Package | Min Version | Role |
|---|---|---|
| `matplotlib` | ≥3.7.0 | Plotting training curves, prediction overlays |
| `tqdm` | ≥4.65.0 | Progress bars in training loops |
| `tensorboard` | (optional, commented out) | TensorBoard integration via `torch.utils.tensorboard.SummaryWriter` in `sam3/train/utils/logger.py` |
| `wandb` | (optional, commented out) | Weights & Biases support stub in `sam3/train/trainer.py` (field `wandb_writer`) |

### Video & Multimodal
| Package | Min Version | Role |
|---|---|---|
| `decord` | ≥0.6.0 | GPU-accelerated video decoding |
| `open-clip-torch` | ≥2.20.0 | OpenCLIP text/vision encoder used in multimodal components |
| `ftfy` | ≥6.1.0 | Text normalization for CLIP tokenization |
| `regex` | ≥2023.0.0 | Regex utilities for text processing |

### Job Scheduling
| Package | Min Version | Role |
|---|---|---|
| `submitit` | ≥1.4.0 | SLURM job array submission for native SAM3 training (`sam3/train/`) |
| `iopath` | ≥0.1.10 | Pathmanager (`g_pathmgr`) for checkpoints and file I/O |

### Performance / GPU Kernels
| Library | Role |
|---|---|
| `triton` | Custom Triton JIT kernels for sigmoid focal loss (`sam3/train/loss/sigmoid_focal_loss.py`) |
| `flash_attn_interface` (FA3) | FlashAttention 3 (FP8) used optionally in `sam3/sam/transformer.py` via `sam3/perflib/fa3.py` |

---

## Key Dependencies

**Critical path (must be present to run):**
- `torch>=2.7.0` — everything depends on this
- `transformers>=4.48.0` — `Sam3Model` / `Sam3Processor` APIs
- `huggingface-hub>=0.26.0` — model download from HuggingFace Hub (`facebook/sam3`)
- `Pillow>=10.0.0`, `numpy>=1.24.0`, `opencv-python>=4.8.0` — image pipeline
- `pycocotools>=2.0.6` — annotation loading and IoU evaluation
- `hydra-core>=1.3.0`, `omegaconf>=2.3.0` — config for native SAM3 training path
- `PyYAML>=6.0` — config for standalone LoRA training path

**Optional / enhancement:**
- `wandb` — experiment tracking (commented out in `requirements.txt`)
- `tensorboard` — training visualization (commented out in `requirements.txt`, code support present)
- `flash_attn_interface` — FA3 high-performance attention (runtime optional, not in requirements.txt)
- `triton` — focal loss GPU kernels (runtime optional in loss_fns.py)

---

## Configuration

**YAML-based configs (`sam3_lora_configs/`):**
- `base_config.yaml` — baseline LoRA training config (rank=8, α=16, fp16, cosine LR)
- `full_lora_config.yaml` — all components enabled
- `light_lora_config.yaml` — minimal LoRA for low-VRAM
- `minimal_lora_config.yaml` — smallest possible footprint
- `crack_detection_config.yaml` — domain-specific preset
- `sam3_lora_standalone.yaml` — standalone mode

**Hydra configs (`sam3/train/configs/`):**
- `roboflow_v100/` — Roboflow-100 dataset fine-tuning recipes
- `odinw13/` — ODinW-13 benchmark configs
- `gold_image_evals/`, `silver_image_evals/`, `saco_video_evals/` — evaluation presets

**Hardware config (from `base_config.yaml`):**
- `device: cuda` (falls back to CPU)
- `mixed_precision: fp16` (fp16/bf16/no supported)
- `use_compile: false` (torch.compile opt-in)
- Multi-GPU: `torchrun`-based DDP, triggered by `--device 0 1 2 3` CLI flag

---

**Dev Dependencies:**
- `pytest>=7.4.0` — unit tests (`test_lora_injection.py`, `test_aux_outputs.py`)
- `black>=23.3.0` — code formatting
- `flake8>=6.0.0` — linting

---

## Platform Requirements

**Development:**
- Python ≥3.8 (3.12.3 confirmed working)
- CUDA GPU (16GB+ VRAM recommended for LoRA training)
- Linux (Bash scripts; SLURM optional for distributed runs)

**Production / Inference:**
- CUDA GPU for full-speed inference
- CPU fallback supported but slow
- No containerisation or Docker config detected
- Checkpoints: LoRA-only `.pt` files (10–50 MB) or full model via HuggingFace `save_pretrained`

---

*Stack analysis: 2025-07-14*
