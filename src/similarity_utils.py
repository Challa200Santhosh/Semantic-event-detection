# Handles cosine similarity computation

import torch
import torch.nn.functional as F

def compute_similarity(image_features, text_features, metric="cosine"):
    """
    Compute similarity between image embeddings and text embeddings.
    
    Args:
        image_features (torch.Tensor): Image embeddings (batch_size, embed_dim)
        text_features (torch.Tensor): Text embeddings (num_prompts, embed_dim)
        metric (str): Similarity metric ("cosine", "dot", "euclidean")
        
    Returns:
        torch.Tensor: Similarity scores (batch_size, num_prompts)
    """
    
    if metric == "cosine":
        # Normalize vectors (important for cosine similarity)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity score
        similarity = image_features @ text_features.T
        
    elif metric == "dot":
        # Direct dot product (assumes pre-normalized features in CLIP)
        similarity = image_features @ text_features.T
        
    elif metric == "euclidean":
        # Euclidean distance (converted to similarity)
        distance = torch.cdist(image_features, text_features, p=2)
        similarity = -distance  # Negative distance as similarity
        
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarity
