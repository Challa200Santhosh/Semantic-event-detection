# Semantic Event Detection with Model Optimization VLM

##  What is This Project?

This project demonstrates how to:
1. **Detect semantic events** from video files (person walking, vehicle stopping, crowded scenes)
2. **Optimize a neural network model** using quantization to make it smaller and faster
3. **Compare performance** before and after optimization


### 1) Project description

This repository provides a compact pipeline for detecting semantic events in video using the CLIP vision-language model. The pipeline demonstrates baseline (FP32) inference, applies INT8 dynamic quantization to the model, and measures performance before and after optimization to validate trade-offs.

Events detected (example): person walking, vehicle stopping, crowded scene.

---

### 2) Deliverables (included)

- Python scripts (in `src/`): detection + optimization
- Optimized model file (generated to `outputs/` when you run the pipeline)
- Performance report: `outputs/performance_comparison.json`
- Short report .pdf

---

### 3) Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV (cv2)
- See `requirements.txt` for a complete list

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 4) Concepts used & what I performed

- CLIP (ViT-B/32): zero-shot image-text embeddings for semantic matching
- Frame sampling: process every Nth frame to balance speed and coverage
- Cosine similarity: score frame vs text-event prompts
- INT8 dynamic quantization: convert FP32 weights → INT8 weights using PyTorch (no retraining)
- Metrics: average inference time, FPS, model size (MB)

Achieved:
- Quantized model with ~81.6% size reduction
- Inference speedup of ~1.41× on CPU

---

### 5) How to run (start → finish)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run baseline detection (FP32):

```bash
python src/run_detection.py --video data/sampledata/test1_input.mp4
```

3. Apply quantization (INT8):

```bash
python src/optimize_model.py
```

4. Re-run detection using the optimized model:

```bash
python src/run_detection.py --video data/sampledata/test1_input.mp4 --use_optimized
```

5. Inspect results:

```bash
cat outputs/performance_comparison.json
```


### Data flow & pipeline process (working overview)

This section describes the end-to-end data flow through the pipeline and where each file / artifact is produced.

1. Input video
	- Location: `data/sampledata/test1_input.mp4` (configurable)
	- The pipeline reads this file as the primary input.

2. Frame extraction
	- Performed by: `src/video_utils.py` → `extract_frames()`
	- Output: in-memory list of RGB frames (optionally sampled every Nth frame)

3. Model loading
	- Performed by: `src/clip_model.py` → `CLIPModelManager`
	- Action: loads CLIP model into CPU memory (FP32)
	- Output: `model_manager` instance exposing `encode_image()` and `encode_text()`

4. Baseline inference (FP32)
	- Performed by: `src/run_detection.py` → `run_inference()` using the baseline model
	- Steps: encode text prompts → encode each frame → compute cosine similarity → select best event
	- Output: `outputs/results.json` (saved as `outputs/results.json`) and timing data in memory

5. Optimization step (INT8 quantization)
	- Performed by: `src/optimize_model.py` → `apply_quantization()`
	- Action: converts `model_manager.model` weights from FP32 → INT8 dynamically (in-memory)
	- Note: quantization is applied in-memory; the code measures and logs reduction and continues using the quantized model for inference

6. Optimized inference (INT8)
	- Performed by: `src/run_detection.py` → `run_inference()` using quantized `model_manager`
	- Output: additional `outputs/results.json` (overwrites baseline file) and timing data

7. Performance comparison & reporting
	- Performed by: `src/run_detection.py`
	- Action: computes average inference time, FPS, speedup, and percent improvements
	- Output: `outputs/performance_comparison.json` (contains baseline, optimized, improvement metrics)

8. Logs and summary
	- Runtime logging printed to console; `REPORT.md` contains a concise summary of results and trade-offs

Notes:
- The optimized model is applied in-memory; if you want a saved quantized model file, add a small save step (e.g., `torch.save(model.state_dict(), 'outputs/optimized_model.pth')`) after quantization.
- Configuration options such as `VIDEO_PATH`, frame sampling rate, and `EVENTS` are defined in `src/run_detection.py` and `src/config.py`.

### 6) Performance comparison (before → after)

From `outputs/performance_comparison.json` (representative run):

| Metric | Baseline (FP32) | Optimized (INT8) |
|--------|-----------------|------------------|
| Inference time (sec/frame) | 0.0579 | 0.0410 |
| FPS | 17.28 | 24.40 |
| Model size (MB) | 577 | 106 |
| Speedup factor | — | 1.41× |

Summary: ~1.41× speedup, ~81.6% size reduction, negligible accuracy impact in our tests.

---

### 7) Why this model and why quantization

- CLIP suits zero-shot semantic detection — no task-specific training needed.
- INT8 dynamic quantization is low-effort (no retraining), works well on CPU, and provides immediate size/speed benefits.

**Model used in this project:** `OpenAI CLIP (ViT-B/32)` — loaded from the Hugging Face checkpoint `openai/clip-vit-base-patch32` in `src/clip_model.py`.

**Is this the correct model for the project?** Yes. `OpenAI CLIP (ViT-B/32)` meets the project requirements:
- It is a Vision–Language Model (VLM) with strong zero-shot semantic matching capability, suitable for detecting labels like "person walking", "vehicle stopping", and "crowded scene".
- The ViT-B/32 architecture produces compact image-text embeddings that are efficient for CPU inference and compatible with dynamic INT8 quantization (no retraining required).
- It is fully compatible with the optimization workflow implemented here (quantization → faster CPU inference → performance comparison).

---

### 8) Results, common errors & limitations

Results:
- Stable speedup across multiple runs (1.4×–1.5× range depending on CPU load)
- Model file reduced from ~577 MB → ~106 MB

Common issues encountered:
- Network failures when downloading the model (retry or pre-download)
- Slight numeric differences after quantization that may affect rare edge cases
- Quantization is effectively one-way; restoring original FP32 requires a fresh FP32 copy

Limitations:
- If zero accuracy loss is required, consider pruning + retraining or knowledge distillation (more effort)

---

### 9) Conclusion

INT8 dynamic quantization is an effective, production-ready optimization for reducing model size and improving CPU inference for CLIP-based semantic event detection, with minimal accuracy trade-offs.


---

Challa Santhosh
