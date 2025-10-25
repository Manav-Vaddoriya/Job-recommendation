import torch
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple
from models.neural_network import JobDomainClassifier

class DomainPredictor:
    """Handles job domain prediction using the trained model."""
    
    def __init__(self, model_path: str, domain_embed_map_path: str):
        self.model = None
        self.label_encoder = None
        self.domain_embed_map = None
        self.load_model(model_path, domain_embed_map_path)
    
    def load_model(self, model_path: str, domain_embed_map_path: str):
        """Load the trained model and domain mappings."""
        self.domain_embed_map = pickle.load(open(domain_embed_map_path, "rb"))
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        self.model = JobDomainClassifier(
            input_dim=1024,
            hidden_dim=256,
            num_classes=len(self.domain_embed_map),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = checkpoint["label_encoder"]
    
    def predict_topk(self, embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Predict top-k job domains."""
        self.model.eval()
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(embedding)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_idxs = torch.topk(probs, k=k)

        top_probs = top_probs.cpu().numpy().flatten()
        top_idxs = top_idxs.cpu().numpy().flatten()
        top_domains = self.label_encoder.inverse_transform(top_idxs)
        return list(zip(top_domains, top_probs))