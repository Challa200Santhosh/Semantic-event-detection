# Handles loading CLIP model and encoding

import torch
import logging
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class CLIPModelManager:
    """
    Manages CLIP model loading and inference for semantic event detection.
    """
    
    def __init__(self, device="cpu"):
        """
        Initialize CLIP model and processor.
        
        Args:
            device (str): Device to use ("cpu" or "cuda")
        """
        # Set device (CPU for resource-limited systems)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load pretrained CLIP model
            logger.info("Loading CLIP model from openai/clip-vit-base-patch32...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # Processor handles preprocessing (resize, normalize etc.)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Set model to evaluation mode
            self.model.eval()
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise

    def encode_text(self, text_list):
        """
        Convert text prompts to embeddings.
        
        Args:
            text_list (list): List of text prompts
            
        Returns:
            torch.Tensor: Text embeddings (normalized)
        """
        try:
            inputs = self.processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_output = self.model.text_model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
                text_features = self.model.text_projection(text_output.pooler_output)
            
            return text_features
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def encode_image(self, image):
        """
        Convert image to embedding.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Image embedding (normalized)
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_output = self.model.vision_model(**inputs)
                image_features = self.model.visual_projection(image_output.pooler_output)
            
            return image_features
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise
