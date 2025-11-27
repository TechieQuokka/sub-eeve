"""
EEVE-Korean model loader using llama-cpp-python
"""
from pathlib import Path
from llama_cpp import Llama
from config import MODEL_PATH, N_CTX, N_GPU_LAYERS, N_BATCH, VERBOSE


def load_model(model_path: str = None) -> Llama:
    """
    Load EEVE-Korean GGUF model using llama.cpp.

    Args:
        model_path: Optional custom model path. Uses config.MODEL_PATH if not provided.

    Returns:
        Loaded Llama model instance

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if model_path is None:
        model_path = MODEL_PATH

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please download EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf to {path.parent}"
        )

    print(f"Loading EEVE-Korean model from: {model_path}")
    print(f"Configuration: ctx={N_CTX}, gpu_layers={N_GPU_LAYERS}, batch={N_BATCH}")

    try:
        model = Llama(
            model_path=str(path),
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=N_BATCH,
            logits_all=True,  # Enable logprobs support for best-of-N sampling
            verbose=VERBOSE
        )
        print("✓ Model loaded successfully (with logprobs support)")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
