import time
import json
import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
import logging

from clip_model import CLIPModelManager
from video_utils import extract_frames
from similarity_utils import compute_similarity
from optimize_model import apply_quantization

# -----------------------
# Logging Configuration
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------
# Configuration
# -----------------------
VIDEO_PATH = "data/sampledata/test1_input.mp4"
OUTPUT_PATH = "outputs/results.json"
COMPARISON_PATH = "outputs/performance_comparison.json"

EVENTS = [
    "person walking",
    "vehicle stopping",
    "crowded scene"
]

DEVICE = "cpu"  # CPU for resource-limited system


def run_inference(model_manager, model_variant="baseline"):
   
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Video file not found: {VIDEO_PATH}")
        sys.exit(1)
    
    # Extract frames
    logger.info(f"Extracting frames from {VIDEO_PATH}...")
    frames = extract_frames(VIDEO_PATH)
    
    if not frames:
        logger.error("No frames extracted from video")
        sys.exit(1)
    
    logger.info(f"Extracted {len(frames)} frames")

    # Encode text prompts once
    logger.info("Encoding text prompts...")
    text_features = model_manager.encode_text(EVENTS)

    results = []
    inference_times = []

    logger.info(f"Running {model_variant} model inference...")
    for idx, frame in enumerate(tqdm(frames, desc=f"{model_variant} inference")):
        try:
            image = Image.fromarray(frame)

            start_time = time.time()

            image_features = model_manager.encode_image(image)
            similarity = compute_similarity(image_features, text_features)

            end_time = time.time()

            # Measure inference time
            inference_times.append(end_time - start_time)

            # Get best matching event
            best_idx = similarity.argmax().item()
            event = EVENTS[best_idx]
            confidence = float(similarity[0][best_idx].item())

            results.append({
                "frame": idx,
                "event": event,
                "confidence": confidence,
                "similarity_scores": {EVENTS[i]: float(similarity[0][i].item()) for i in range(len(EVENTS))}
            })
        except Exception as e:
            logger.error(f"Error processing frame {idx}: {str(e)}")
            continue

    if not inference_times:
        logger.error("No inference times recorded")
        return results, []
    
    avg_time = sum(inference_times) / len(inference_times)
    fps = 1 / avg_time

    logger.info(f"\n{'='*60}")
    logger.info(f"Results ({model_variant.upper()})")
    logger.info(f"{'='*60}")
    logger.info(f"Total Frames Processed: {len(results)}")
    logger.info(f"Average Inference Time: {avg_time:.4f} seconds")
    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"{'='*60}\n")

    return results, inference_times


def main():
   
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        
        logger.info("="*60)
        logger.info("Semantic Event Detection Pipeline")
        logger.info("="*60)
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Events to detect: {', '.join(EVENTS)}")
        logger.info("="*60)
        
        # ===== BASELINE MODEL =====
        logger.info("\n[1/3] Loading baseline CLIP model...")
        baseline_start_time = time.time()
        model_manager = CLIPModelManager(device=DEVICE)
        baseline_load_time = time.time() - baseline_start_time
        logger.info(f"Model loaded in {baseline_load_time:.4f} seconds")

        logger.info("\n[2/3] Running baseline inference...")
        baseline_results, baseline_times = run_inference(model_manager, "baseline")
        baseline_avg_time = sum(baseline_times) / len(baseline_times) if baseline_times else 0
        baseline_fps = 1 / baseline_avg_time if baseline_avg_time > 0 else 0

        # Save baseline results
        with open(OUTPUT_PATH, "w") as f:
            json.dump(baseline_results, f, indent=4)
        logger.info(f"Baseline results saved to {OUTPUT_PATH}")

        # ===== OPTIMIZED MODEL =====
        logger.info("\n[3/3] Applying quantization optimization...")
        opt_start_time = time.time()
        model_manager.model = apply_quantization(model_manager.model)
        opt_apply_time = time.time() - opt_start_time
        logger.info(f"Optimization applied in {opt_apply_time:.4f} seconds")

        logger.info("\nRunning optimized inference...")
        optimized_results, optimized_times = run_inference(model_manager, "optimized")
        optimized_avg_time = sum(optimized_times) / len(optimized_times) if optimized_times else 0
        optimized_fps = 1 / optimized_avg_time if optimized_avg_time > 0 else 0

        # ===== PERFORMANCE COMPARISON =====
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*60)
        
        speedup = baseline_avg_time / optimized_avg_time if optimized_avg_time > 0 else 1
        fps_improvement = ((optimized_fps - baseline_fps) / baseline_fps * 100) if baseline_fps > 0 else 0
        time_reduction = ((baseline_avg_time - optimized_avg_time) / baseline_avg_time * 100) if baseline_avg_time > 0 else 0
        
        logger.info(f"{'Metric':<30} {'Baseline':<20} {'Optimized':<20} {'Improvement':<15}")
        logger.info("-" * 85)
        logger.info(f"{'Avg Inference Time (s)':<30} {baseline_avg_time:<20.4f} {optimized_avg_time:<20.4f} {time_reduction:.2f}%")
        logger.info(f"{'FPS':<30} {baseline_fps:<20.2f} {optimized_fps:<20.2f} {fps_improvement:+.2f}%")
        logger.info(f"{'Speedup Factor':<30} {'1.0x':<20} {f'{speedup:.2f}x':<20}")
        logger.info("="*60)

        # Save comparison
        comparison = {
            "baseline": {
                "avg_inference_time": baseline_avg_time,
                "fps": baseline_fps,
                "model_load_time": baseline_load_time,
                "total_frames": len(baseline_results)
            },
            "optimized": {
                "avg_inference_time": optimized_avg_time,
                "fps": optimized_fps,
                "optimization_time": opt_apply_time,
                "total_frames": len(optimized_results)
            },
            "improvement": {
                "speedup_factor": speedup,
                "fps_improvement_percent": fps_improvement,
                "time_reduction_percent": time_reduction
            }
        }
        
        with open(COMPARISON_PATH, "w") as f:
            json.dump(comparison, f, indent=4)
        logger.info(f"\nComparison saved to {COMPARISON_PATH}")
        
        # Print event summary
        logger.info("\n" + "="*60)
        logger.info("DETECTED EVENTS SUMMARY (Baseline Model)")
        logger.info("="*60)
        event_counts = {}
        for result in baseline_results:
            event = result["event"]
            event_counts[event] = event_counts.get(event, 0) + 1
        
        for event, count in event_counts.items():
            percentage = (count / len(baseline_results)) * 100
            logger.info(f"{event:<30} {count:<10} ({percentage:.1f}%)")
        logger.info("="*60)
        
        logger.info("\n✓ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
