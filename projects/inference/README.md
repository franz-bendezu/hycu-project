# Vision Inference Service

High-performance furniture and component detection service using YOLO11 (ONNX) and FastAPI.

## Project Structure

The service follows a modular architecture for maintainability and scalability:

```text
inference/
├── app/
│   ├── core/           # Configuration and domain constants
│   ├── services/       # Business logic (Detector, Processing)
│   ├── utils/          # Image processing and helper utilities
│   ├── schemas.py      # Pydantic request/response models
│   └── main.py         # FastAPI entry point & lifespan management
├── models/             # Local storage for .onnx artifacts
├── tests/              # Pytest suite
├── Dockerfile          # Container definition
└── requirements.txt    # GPU-enabled dependencies
```

## Hardware & Performance

This service is optimized for **GPU inference** but includes a **Safe Fallback** for CPU-only environments.

- **GPU Support**: Uses `onnxruntime-gpu` for native acceleration on NVIDIA GPUs (like RTX 5080).
- **Safe Fallback**: If CUDA drivers or GPU hardware are not detected, the service automatically falls back to the `CPUExecutionProvider` without crashing.
- **Dynamic Selection**: Hardware priority is controlled via environment variables.

### Performance Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_PROVIDERS` | `CUDAExecutionProvider,CPUExecutionProvider` | Comma-separated list of fallback priority |
| `INFERENCE_CONFIDENCE_THRESHOLD` | `0.10` | Min confidence for component detection |
| `INFERENCE_IMAGE_SIZE` | `640` | Internal model resolution |
| `INFERENCE_SEGMENTATION_BACKEND` | `sam2` | Segmentation backend (`sam2` or `box-rasterizer`) |
| `INFERENCE_SAM2_MODEL_PATH` | `` | Path to SAM2 checkpoint/model used for component masks |
| `INFERENCE_SAM_SEARCH_MODE` | `beam` | SAM fallback optimizer mode (`beam` or `rjmcmc`) |
| `INFERENCE_SAM_BEAM_WIDTH` | `12` | Beam width used when `INFERENCE_SAM_SEARCH_MODE=beam` |
| `INFERENCE_SAM_DEPTH_WEIGHT` | `0.0` | Optional depth term weight in SAM proposal scoring |
| `INFERENCE_SAM_DEBUG_DIR` | `app/debug/sam2` | Output directory for SAM debug artifacts |

Note: segmentation now defaults to `sam2`. If SAM2 is unavailable (missing model/runtime), the service automatically falls back to `box-rasterizer`.

### SAM Fallback Optimizer Modes

Use deterministic mode by default:

```bash
export INFERENCE_SAM_SEARCH_MODE=beam
```

Enable stochastic refinement (rjMCMC-style local search):

```bash
export INFERENCE_SAM_SEARCH_MODE=rjmcmc
```

Example startup with explicit optimizer flags:

```bash
cd /workspaces/hycu-project/projects/inference
export INFERENCE_SEGMENTATION_BACKEND=sam2
export INFERENCE_SAM2_MODEL_PATH=/workspaces/hycu-project/projects/inference/models/sam2_hiera_tiny.pt
export INFERENCE_SAM_SEARCH_MODE=rjmcmc
export INFERENCE_SAM_BEAM_WIDTH=12
export INFERENCE_SAM_DEPTH_WEIGHT=0.0
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

## Installation & Setup

### Requirements

- Python 3.11+
- NVIDIA Drivers + CUDA/cuDNN (for GPU acceleration)

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the service:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload
   ```

### Docker (Recommended)

To run with GPU support, ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed:

```bash
docker build -t furniture-inference .
docker run --gpus all -p 9000:9000 furniture-inference
```

## API Endpoints

### 1. Health Check
`GET /health`
Returns service status, model path, and the **active hardware provider**.

### 2. Batch Inference
`POST /infer`
Processes one or more image URLs (or Base64 data URLs) and returns an aggregated furniture analysis including:
- Detected product type (`cabinet`, `desk`, `shelf`).
- Structural components (panels, legs).
- Hardware recommendations (hinges, slides, screws).
- Estimated physical dimensions.

## Testing

Run the test suite to verify the modular structure and fallback logic:
```bash
PYTHONPATH=$PYTHONPATH:. pytest tests/test_main.py
```
