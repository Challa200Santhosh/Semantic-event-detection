# Configuration file for semantic event detection pipeline

# ========================
# Video Configuration
# ========================

# Path to input video file
VIDEO_PATH = "data/sampledata/test1_input.mp4"

# Extract every nth frame (higher = faster processing, lower = more detailed)
# frame_skip=10 means process 1 out of every 10 frames
FRAME_SKIP = 10

# ========================
# Output Configuration
# ========================

# Directory for output files
OUTPUT_DIR = "outputs"

# Results file name
RESULTS_FILE = "results.json"

# Performance comparison file name
COMPARISON_FILE = "performance_comparison.json"

# ========================
# Event Configuration
# ========================

# Semantic events to detect
# Add or modify events as needed
EVENTS = [
    "person walking",
    "vehicle stopping",
    "crowded scene"
]

# ========================
# Model Configuration
# ========================

DEVICE = "cpu"

# CLIP model variant
# Options: "openai/clip-vit-base-patch32" (smaller, faster)
#          "openai/clip-vit-large-patch14" (larger, more accurate)
MODEL_NAME = "openai/clip-vit-base-patch32"

# ========================
# Optimization Configuration
# ========================

# Apply quantization to optimize model
APPLY_QUANTIZATION = True

# Optional: Apply pruning (experimental)
APPLY_PRUNING = False
PRUNING_AMOUNT = 0.3  # Remove 30% of weights

# ========================
# Logging Configuration
# ========================

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Save logs to file
SAVE_LOGS = True
LOG_FILE = "outputs/detection.log"

# ========================
# Inference Configuration
# ========================

# Confidence threshold for events (0.0 to 1.0)
# Only report detections above this threshold
CONFIDENCE_THRESHOLD = 0.0  # Report all detections

# Similarity metric: "cosine", "dot", "euclidean"
SIMILARITY_METRIC = "cosine"
