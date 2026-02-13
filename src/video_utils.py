# Handles video reading and frame extraction

import cv2
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path, frame_skip=10):
   
    import os
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    frames = []
    frame_count = 0
    extracted_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every nth frame
            if frame_count % frame_skip == 0:
                # Convert BGR to RGB (CLIP expects RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                extracted_count += 1

            frame_count += 1
        
        logger.info(f"Successfully extracted {extracted_count} frames from {frame_count} total frames (skip={frame_skip})")
        
    finally:
        cap.release()
    
    return frames
