"""
EEVE-Korean Subtitle Translator Configuration
"""
from pathlib import Path

# Model Configuration
MODEL_PATH = "/home/beethoven/workspace/deeplearning/project/autokr2/polyglot-sub-pipeline/models/EEVE-Korean-Instruct-10.8B-v1.0.Q8_0.gguf"

# Model Parameters
N_CTX = 8192              # Context window size (supports up to 32K but 8K is efficient)
N_GPU_LAYERS = -1         # -1 = use all GPU layers (RTX 3060 12GB can handle Q8_0)
N_BATCH = 512             # Batch size for processing
TEMPERATURE = 0.3         # Lower = more deterministic, higher = more creative
MAX_TOKENS = 512          # Maximum tokens per translation

# Translation Configuration
CHUNK_SIZE = 20           # Number of segments to translate at once
CONTEXT_BEFORE = 10       # Number of previous segments for context
CONTEXT_AFTER = 5         # Number of following segments for context

# Best-of-N Sampling Configuration
BEST_OF_N = 3             # Number of candidates to generate (1 = disabled, 3-5 recommended)
SAMPLING_TEMPERATURE = 0.7  # Temperature for candidate generation (higher = more diverse)

# Language Configuration
LANGUAGE_MAP = {
    'ja': '일본어',
    'en': '영어',
    'ko': '한국어',
    'zh': '중국어'
}

# Field name mapping for target languages
FIELD_NAME_MAP = {
    'ko': 'text_ko',
    'en': 'text_en',
    'ja': 'text_ja',
    'zh': 'text_zh'
}

# Progress Configuration
SHOW_PROGRESS = True
VERBOSE = False
