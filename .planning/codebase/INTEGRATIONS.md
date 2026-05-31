# Integrations

**Analysis Date:** 2025-07-14

---

## External APIs & Services

### Hugging Face Hub
- **Purpose:** Pre-trained model weights download and (optionally) checkpoint upload
- **SDK/Client:** `huggingface-hub>=0.26.0`, `transformers>=4.48.0`
- **Usage:**
  - `Sam3Model.from_pretrained("facebook/sam3")` in `train_sam3_lora.py` (line 235)
  - `Sam3Processor.from_pretrained("facebook/sam3")` in `train_sam3_lora.py` (line 239)
  - `model.save_pretrained(...)` / `processor.save_pretrained(...)` for checkpoint saving
- **Auth:** HuggingFace token via `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` env vars (standard `huggingface-hub` mechanism)
- **Config toggle:** `output.push_to_hub: false` and `output.hub_model_id: null` in `sam3_lora_configs/base_config.yaml`

### OpenAI API (LLM Agent)
- **Purpose:** Vision-language agent feature — sends images and prompts to an LLM for guided segmentation queries
- **SDK/Client:** `openai>=1.0.0`
- **Usage:** `sam3/agent/client_llm.py` — `from openai import OpenAI`; `send_generate_request()` function targets `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` by default (OpenAI-compatible endpoint)
- **Auth:** `api_key` parameter passed to `OpenAI()`; `server_url` configurable to point at custom or local OpenAI-compatible servers
- **Note:** Default model name suggests a self-hosted or Together.ai/Fireworks-style proxy, not necessarily `api.openai.com`

---

## Databases & Storage

### Local Filesystem (Primary)
- All training data, checkpoints, and outputs stored on local disk
- No cloud object storage (S3, GCS, Azure Blob) integration detected in application code
- Dataset root configured via `paths.dataset_root` in `loral.yaml` and `paths.roboflow_vl_100_root` in Hydra configs

### iopath PathManager (`g_pathmgr`)
- **Package:** `iopath>=0.1.10`
- **Usage:** `sam3/train/utils/checkpoint_utils.py`, `sam3/train/trainer.py`, `sam3/train/utils/train_utils.py`
- Abstracts file I/O; supports local paths and pluggable backends (HTTP, manifold, S3 via iopath handlers — none registered by default)
- All checkpoint reads/writes go through `g_pathmgr`

### Checkpointing
- **LoRA-only checkpoints:** `lora_weights.pt` — lightweight (10–50 MB), saved via `torch.save`
- **Full model checkpoints:** `model.safetensors` + `processor_config.json` via `save_pretrained`
- **Native trainer checkpoints:** `checkpoint.pt` managed by `sam3/train/utils/checkpoint_utils.py`
- Location: `outputs/sam3_lora/<checkpoint_name>/` (configurable via `output.output_dir`)

---

## Third-Party Libraries (Key)

### COCO API (`pycocotools`)
- **Purpose:** Parse COCO-format annotation files (`_annotations.coco.json`) for training and evaluation
- **Usage:** Dataset loading in `sam3/train/data/`, evaluation metrics (IoU, AP)
- **Data format conversion:** `convert_roboflow_to_coco.py` converts Roboflow per-image JSON exports to COCO format

### Roboflow Dataset Format
- **Source:** Roboflow-exported datasets (COCO JSON format)
- **Conversion:** `convert_roboflow_to_coco.py` — converts Roboflow per-image JSON to a single `_annotations.coco.json`
- **Native support:** `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` contains full Roboflow-100 training pipeline

### OpenCLIP (`open-clip-torch`)
- **Purpose:** Text encoder backbone used inside SAM3's multimodal pipeline
- **Version:** ≥2.20.0
- **Usage:** Referenced in model architecture; BPE vocabulary file at `assets/bpe_simple_vocab_16e6.txt.gz` is required at runtime (path set in `loral.yaml` and Hydra configs as `paths.bpe_path`)

### Triton JIT Kernels
- **Purpose:** High-performance GPU implementation of sigmoid focal loss
- **Files:** `sam3/train/loss/sigmoid_focal_loss.py` — custom `@triton.jit` kernels
- **Usage:** `sam3/train/loss/loss_fns.py` calls `triton_sigmoid_focal_loss` / `triton_sigmoid_focal_loss_reduce`
- **Fallback:** `loss_fns.py` has a `triton=True/False` flag; PyTorch fallback available

### FlashAttention 3 (`flash_attn_interface`)
- **Purpose:** FP8 FlashAttention for high-throughput inference/training on H100 GPUs
- **Files:** `sam3/perflib/fa3.py` wraps `flash_attn_interface.flash_attn_func` as a `torch.library.custom_op`
- **Usage:** `sam3/sam/transformer.py` (lines 248, 343) — optionally invoked in attention layers
- **Requirement:** Not in `requirements.txt`; must be installed separately if needed

### HOTA & TETA Evaluation Toolkits
- **Purpose:** Multi-object tracking evaluation metrics
- **Files:**
  - `sam3/eval/hota_eval_toolkit/trackeval/` — HOTA metric implementation
  - `sam3/eval/teta_eval_toolkit/metrics/` — TETA metric implementation
- **Usage:** Evaluation pipelines for video/tracking tasks

### submitit (SLURM Integration)
- **Purpose:** SLURM job array submission for large-scale distributed training
- **Version:** ≥1.4.0
- **Usage:** `sam3/train/configs/roboflow_v100/` Hydra configs reference `submitit.job_array.task_index` for dataset sharding
- **Scope:** Native SAM3 training path (`sam3/train/`) only; not used in standalone LoRA scripts

---

## Monitoring & Observability

### TensorBoard
- **Status:** Optional (commented out in `requirements.txt`); code support present
- **Integration:** `sam3/train/utils/logger.py` — `make_tensorboard_logger()` using `torch.utils.tensorboard.SummaryWriter`
- **Config:** `logging.tensorboard_writer._target_: sam3.train.utils.logger.make_tensorboard_logger`; log dir at `${launcher.experiment_log_dir}/tensorboard`
- **Activation:** Enabled in all `sam3/train/configs/` YAML files

### Weights & Biases (wandb)
- **Status:** Optional (commented out in `requirements.txt`); stub present in trainer
- **Integration:** `sam3/train/trainer.py` — field `wandb_writer: Optional[Any] = None` (line 142)
- **Config:** `wandb_writer: null` in all Hydra config files (disabled by default)

### Console Logging
- **Primary logging:** Python `logging` module + `tqdm` progress bars
- **Setup:** `sam3/train/utils/logger.py` — `setup_logging()` function
- **Standalone scripts:** Direct `print()` statements in `train_sam3_lora.py`

---

## Cloud & Infrastructure

### Distributed Training (Multi-GPU)
- **Backend:** `torch.distributed` with NCCL backend (`sam3/train/utils/distributed.py`)
- **Launch method:** `torchrun` (auto-launched by training scripts when `--device 0 1 2 3` is specified)
- **Wrapper:** `sam3/train/utils/train_utils.py` — `init_distributed_mode()` initialises process group
- **DDP:** `torch.nn.parallel.DistributedDataParallel` used in native SAM3 trainer

### SLURM (HPC Clusters)
- **Integration:** `submitit>=1.4.0`
- **Scope:** Native SAM3 training configs (`sam3/train/`) reference SLURM job arrays
- **Standalone LoRA scripts** (`train_sam3_lora.py`, `train_sam3_lora_native.py`) do not use submitit

### No Cloud Provider SDK Detected
- No AWS SDK (`boto3`), Google Cloud SDK, or Azure SDK imports found
- All storage is local filesystem via `iopath`
- Deployment target: on-premises GPU server or cloud VM (no managed service integration)

---

## Environment Variables (Required)

| Variable | Purpose |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Authenticate with HuggingFace Hub for model download |
| `OPENAI_API_KEY` (or passed as `api_key`) | OpenAI / OpenAI-compatible LLM API access in `sam3/agent/client_llm.py` |
| `CUDA_VISIBLE_DEVICES` | GPU selection (also controlled via `--device` CLI argument) |

**Note:** No `.env` file committed; environment variables must be set in the shell before running.

---

*Integration audit: 2025-07-14*
