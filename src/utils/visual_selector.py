import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torchvision.transforms as T
from PIL import Image
import open_clip
import scenedetect 

# script to extract timestamps 

class VisualSelector:
    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained_weights: str = "laion2b_s34b_b79k", # https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
        device: Optional[str] = None
    ):
        """
        clip_model_name: base ViT model
        pretrained_weights: specific CLIP weights
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=pretrained_weights, 
            device=self.device
        )
        self.model.eval()
        
    # uses pyscenedetect to find scene transition timestamps (major timestamps)
    # def detect_scene_changes() -> List[float]:
    #     pass
    
    # converts frames into CLIP embeddings
     # https://huggingface.co/docs/hub/open_clip
    @torch.no_grad()
    def extract_frame_embeddings(self, frames: List[Image.Image]) -> torch.Tensor:
        if not frames:
            return torch.empty((0, 512), device=self.device)

        preprocessed_tensors = torch.stack([self.preprocess(frame) for frame in frames]).to(self.device)
        image_features = self.model.encode_image(preprocessed_tensors)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def compute_visual_novelty(self, embeddings: torch.Tensor):
        # compares prior frame to current frame
        num_frames = embeddings.shape[0]
        
        if num_frames <= 1:
            return torch.zeros(num_frames, device=self.device)
        
        # compare 
        all_but_last = embeddings[:-1]
        all_but_first = embeddings[1:]
        similarity = (all_but_last * all_but_first).sum(dim=-1) # cosine similarity but embeddings are already normalized
        
        novelty_scores = 1.0 - similarity
        
        novelty_scores = torch.cat([torch.tensor([0.0], device=self.device), novelty_scores])
        
        return novelty_scores
    
    
    # TODO more functions