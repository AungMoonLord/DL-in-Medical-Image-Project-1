"""
Class Representation and Automated Prototype Filtering Engine.

Solves dataset requirement #1.1:
Automates outlier and defect detection by establishing representative prototypes
(Spatial Medoid templates & Deep Feature Centroid Embeddings) for each Thai character class.
Enables incoming images to be scored, filtered, or flagged for reclassification without manual scanning.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dataset import invert_glyph_if_needed, letterbox_pad
except (ImportError, ValueError):
    from dataset import invert_glyph_if_needed, letterbox_pad


class ClassPrototypeEngine:
    """
    Automated prototype modeling and anomaly detection for Thai character glyphs.
    Supports both Spatial Template (Medoid/SSIM) and Deep Feature Centroid embeddings.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        target_size: Tuple[int, int] = (32, 32),
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.feature_extractor = feature_extractor.to(self.device).eval() if feature_extractor else None

        self.class_prototypes: Dict[int, torch.Tensor] = {} # class_id -> unit embedding centroid
        self.spatial_medoids: Dict[int, np.ndarray] = {}    # class_id -> normalized 32x32 template
        self.class_thresholds: Dict[int, float] = {}        # class_id -> anomaly threshold

    def _preprocess_spatial(self, img: Image.Image) -> np.ndarray:
        """Letterbox and normalize to standardized 0..1 grayscale numpy array (0 = background, 1 = glyph)."""
        inv_img = invert_glyph_if_needed(img)
        padded = letterbox_pad(inv_img, target_size=self.target_size, pad_ratio=0.08, fill_color=(0, 0, 0))
        gray = padded.convert("L")
        arr = np.array(gray, dtype=np.float32) / 255.0
        return arr

    def build_spatial_prototypes(self, dataset_dir: Union[str, Path]) -> Dict[int, np.ndarray]:
        """
        Computes spatial medoid templates for each class directory.
        The medoid is the actual sample with minimum average distance to all other samples.
        """
        dataset_path = Path(dataset_dir)
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]

        for c_dir in sorted(class_dirs, key=lambda d: int(d.name)):
            class_id = int(c_dir.name)
            img_files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png"))
            if not img_files:
                continue

            # Load up to 50 samples for speed
            sample_files = img_files[:50]
            arrays = [self._preprocess_spatial(Image.open(f)) for f in sample_files]

            if len(arrays) == 1:
                self.spatial_medoids[class_id] = arrays[0]
                continue

            # Compute pairwise Euclidean / L1 distances
            stack = np.stack(arrays, axis=0) # [N, H, W]
            flat = stack.reshape(len(arrays), -1)
            dist_matrix = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
            medoid_idx = int(np.argmin(dist_matrix.sum(axis=1)))
            self.spatial_medoids[class_id] = arrays[medoid_idx]

        return self.spatial_medoids

    @torch.no_grad()
    def build_embedding_prototypes(
        self,
        dataset_dir: Union[str, Path],
        model: nn.Module,
        transform: Any,
    ) -> Dict[int, torch.Tensor]:
        """
        Extracts deep feature embeddings and computes the normalized class centroid prototype.
        """
        self.feature_extractor = model.to(self.device).eval()
        dataset_path = Path(dataset_dir)
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]

        for c_dir in sorted(class_dirs, key=lambda d: int(d.name)):
            class_id = int(c_dir.name)
            img_files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png"))
            if not img_files:
                continue

            embeddings = []
            sample_files = img_files[:100]
            for f in sample_files:
                img = Image.open(f).convert("RGB")
                img = invert_glyph_if_needed(img)
                tensor = transform(img).unsqueeze(0).to(self.device)
                
                # Check if model has feature extractor method or forward
                if hasattr(model, "extract_features"):
                    feat = model.extract_features(tensor)
                elif hasattr(model, "features"):
                    feat = model.features(tensor)
                    if hasattr(model, "global_pool"):
                        feat = model.global_pool(feat)
                else:
                    feat = model(tensor)

                feat = feat.view(feat.size(0), -1)
                feat = F.normalize(feat, p=2, dim=-1)
                embeddings.append(feat)

            stacked = torch.cat(embeddings, dim=0) # [N, D]
            centroid = F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=-1) # [1, D]
            self.class_prototypes[class_id] = centroid

            # Compute intra-class cosine similarities to establish anomaly threshold
            sims = (stacked @ centroid.T).squeeze(1).cpu().numpy()
            mean_sim = float(np.mean(sims))
            std_sim = float(np.std(sims))
            # Adaptive 2.5 sigma lower bound
            self.class_thresholds[class_id] = max(0.4, mean_sim - 2.5 * std_sim)

        return self.class_prototypes

    @torch.no_grad()
    def score_sample(
        self,
        img: Image.Image,
        claimed_class_id: int,
        transform: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Scores an incoming image against its claimed class prototype and scans all 72 classes
        to detect potential mislabeling.
        """
        result = {
            "claimed_class": claimed_class_id,
            "is_outlier": False,
            "spatial_similarity": 0.0,
            "embedding_similarity": 0.0,
            "best_matching_class": claimed_class_id,
            "best_matching_similarity": 0.0,
            "recommendation": "Accept",
        }

        # Spatial Template Matching
        if claimed_class_id in self.spatial_medoids:
            medoid = self.spatial_medoids[claimed_class_id]
            curr_arr = self._preprocess_spatial(img)
            # Normalized Cross Correlation
            norm_medoid = (medoid - medoid.mean()) / (medoid.std() + 1e-6)
            norm_curr = (curr_arr - curr_arr.mean()) / (curr_arr.std() + 1e-6)
            ncc = float(np.mean(norm_medoid * norm_curr))
            result["spatial_similarity"] = round(ncc, 4)

        # Deep Embedding Matching
        if self.feature_extractor and self.class_prototypes and transform:
            tensor = transform(img.convert("RGB")).unsqueeze(0).to(self.device)
            if hasattr(self.feature_extractor, "extract_features"):
                feat = self.feature_extractor.extract_features(tensor)
            elif hasattr(self.feature_extractor, "features"):
                feat = self.feature_extractor.features(tensor)
                if hasattr(self.feature_extractor, "global_pool"):
                    feat = self.feature_extractor.global_pool(feat)
            else:
                feat = self.feature_extractor(tensor)

            feat = feat.view(feat.size(0), -1)
            feat = F.normalize(feat, p=2, dim=-1)

            # Compare against all prototypes
            all_classes = sorted(list(self.class_prototypes.keys()))
            all_protos = torch.cat([self.class_prototypes[c] for c in all_classes], dim=0) # [C, D]
            sims = (feat @ all_protos.T).squeeze(0).cpu().numpy() # [C]

            claimed_idx = all_classes.index(claimed_class_id) if claimed_class_id in all_classes else -1
            claimed_sim = float(sims[claimed_idx]) if claimed_idx >= 0 else 0.0
            best_idx = int(np.argmax(sims))
            best_class = all_classes[best_idx]
            best_sim = float(sims[best_idx])

            result["embedding_similarity"] = round(claimed_sim, 4)
            result["best_matching_class"] = best_class
            result["best_matching_similarity"] = round(best_sim, 4)

            threshold = self.class_thresholds.get(claimed_class_id, 0.5)
            if claimed_sim < threshold:
                result["is_outlier"] = True
                if best_class != claimed_class_id and best_sim > threshold:
                    result["recommendation"] = f"Reclassify to Class {best_class} (Is Mislabeled)"
                else:
                    result["recommendation"] = "Exclude / Purge (Is Fragmented / Low Quality)"

        return result
