import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch.nn as nn
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Union


# @TODO clean codes of this section

_model_cache = {}
_model_cache_lock = threading.Lock()
_model_counter = 0  

def get_dino_model(model_size="small", device="cuda"):
    global _model_counter
    cache_key = f"{model_size}_{device}"
    
    with _model_cache_lock:
        if cache_key not in _model_cache:
            _model_counter += 1
            import os
            pid = os.getpid()
            logging.info(f"Process {pid}: Created DINO model #{_model_counter}: {model_size} on {device}")
            _model_cache[cache_key] = DINOScoreCalculator(model_size=model_size, device=device)
        return _model_cache[cache_key]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BaseMetric:
    def __init__(self):
        self.meter = AverageMeter()

    def reset(self):
        self.meter.reset()
        
    def calculate_score(self, batch, update=True):
        """
        Batch: {"gt_im": [PIL Image], "gen_im": [Image]}
        """
        values = []
        batch_size = len(next(iter(batch.values())))
        for index in tqdm(range(batch_size)):
            kwargs = {}
            for key in ["gt_im", "gen_im", "gt_svg", "gen_svg", "caption"]:
                if key in batch:
                    kwargs[key] = batch[key][index]
            try:
                measure = self.metric(**kwargs)
            except Exception as e:
                print("Error calculating metric: {}".format(e))
                continue
            if math.isnan(measure):
                continue
            values.append(measure)

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan")

        score = sum(values) / len(values)
        if update:
            self.meter.update(score, len(values))
            return self.meter.avg, values
        else:
            return score, values

    def metric(self, **kwargs):
        """
        This method should be overridden by subclasses to provide the specific metric computation.
        """
        raise NotImplementedError("The metric method must be implemented by subclasses.")
    
    def get_average_score(self):
        return self.meter.avg

class DINOScoreCalculator(BaseMetric): 
    #@TODO how to make sure DINO always on GPU? check how ray is deliver gpu resources
    def __init__(self, config=None, model_size='large', device='cuda'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.model_size = model_size
        self.model, self.processor = self.get_DINOv2_model(model_size)
        device = device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.device = device

        self.metric = self.calculate_DINOv2_similarity_score

    def get_DINOv2_model(self, model_size):
        if model_size == "small":
            model_size = "facebook/dinov2-small"
        elif model_size == "base":
            model_size = "facebook/dinov2-base"
        elif model_size == "large":
            model_size = "facebook/dinov2-large"
        else:
            raise ValueError(f"model_size should be either 'small', 'base' or 'large', got {model_size}")
        return AutoModel.from_pretrained(model_size), AutoImageProcessor.from_pretrained(model_size)

    def process_input(self, image, processor):
        if isinstance(image, list):
            if all(isinstance(img, Image.Image) for img in image):
                with torch.no_grad():
                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)
                return features
            else:
                features_list = []
                for img in image:
                    features_list.append(self.process_input(img, processor))
                return torch.cat(features_list, dim=0)
        
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(image, torch.Tensor):
            features = image.unsqueeze(0) if image.dim() == 1 else image
        else:
            raise ValueError("Input must be a file path, PIL Image, or tensor of features")
        return features

    def calculate_DINOv2_similarity_score(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('gen_im')
        features1 = self.process_input(image1, self.processor)
        features2 = self.process_input(image2, self.processor)

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(features1, features2).item()
        sim = (sim + 1) / 2

        return sim
    
    def calculate_batch_scores(self, gt_images: List[Any], gen_images: List[Any]) -> List[float]:
        """
        Calculate similarity scores for multiple image pairs in a single batch
        
        Args:
            gt_images: List of ground truth images (PIL Images, file paths, or tensors)
            gen_images: List of generated images (PIL Images, file paths, or tensors)
            
        Returns:
            List of similarity scores (float values between 0-1)
        """      
        if not gt_images: 
            return []
        
        gt_features = self.process_input(gt_images, self.processor)
        
        gen_features = self.process_input(gen_images, self.processor)
        
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(gt_features, gen_features)
        
        scores = [(sim.item() + 1) / 2 for sim in similarities]
        
        return scores
    
    