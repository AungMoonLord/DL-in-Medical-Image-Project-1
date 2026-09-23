"""
Focal Element Identification and Artifact Cleaning Engine.

Solves dataset defect #1.2:
In scanned document bounding-box crops, neighboring characters or punctuation
often leave partial stray stroke fragments along the image borders.
This module uses Connected Component Analysis (CCA) and spatial-focal weighting
to identify and preserve the primary focal character while suppressing peripheral artifacts.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


class FocalElementCleaner:
    """
    Connected-Component-based cleaner that filters out non-destructive
    peripheral bounding-box fragments from adjacent Thai characters.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.015,
        center_decay_sigma: float = 0.4,
        border_margin: int = 1,
        fill_bg_color: int = 255,
    ):
        self.min_area_ratio = min_area_ratio
        self.center_decay_sigma = center_decay_sigma
        self.border_margin = border_margin
        self.fill_bg_color = fill_bg_color

    def clean(
        self,
        image: Union[Image.Image, np.ndarray],
        return_mask: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, np.ndarray]]:
        """
        Cleans image by isolating focal character components and removing border noise.
        """
        # Convert to numpy grayscale
        if isinstance(image, Image.Image):
            pil_mode = image.mode
            np_img = np.array(image.convert("L"))
        else:
            np_img = image.copy()
            if np_img.ndim == 3:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            pil_mode = "L"

        h, w = np_img.shape
        total_pixels = h * w

        # Detect background polarity (Thai dataset is dark glyphs on light bg)
        corners = np.array([np_img[0, 0], np_img[0, -1], np_img[-1, 0], np_img[-1, -1]])
        is_dark_text = np.median(corners) > 100 or np.percentile(np_img, 70) > 100 or np.mean(np_img) > 100

        if is_dark_text:
            # Invert: text becomes white foreground (255), background becomes black (0)
            _, binary = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological close to bridge tiny broken stroke segments
        kernel = np.ones((2, 2), np.uint8)
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_closed, connectivity=8
        )

        if num_labels <= 1:
            # No foreground detected, return original
            res_img = Image.fromarray(np_img).convert("RGB" if pil_mode == "RGB" else "L")
            return (res_img, np.zeros_like(binary)) if return_mask else res_img

        # Center of the image in normalized [0, 1] coordinates
        cx_img = (w - 1) / 2.0
        cy_img = (h - 1) / 2.0
        max_dist = np.sqrt(cx_img**2 + cy_img**2) + 1e-6

        # Score each connected component (exclude background label 0)
        component_scores = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            # Check border touch
            touches_border = (
                bx <= self.border_margin
                or by <= self.border_margin
                or (bx + bw) >= (w - self.border_margin)
                or (by + bh) >= (h - self.border_margin)
            )

            # Spatial distance from image center
            dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2) / max_dist
            spatial_weight = np.exp(-0.5 * (dist / self.center_decay_sigma)**2)

            # Area ratio
            area_weight = area / float(total_pixels)

            # Border penalty: severe if small component touches border
            border_penalty = 0.2 if (touches_border and area_weight < 0.12) else 1.0

            score = area_weight * spatial_weight * border_penalty
            component_scores.append((i, score, area_weight, touches_border, (bx, by, bw, bh), (cx, cy)))

        # Find primary main glyph component
        component_scores.sort(key=lambda x: x[1], reverse=True)
        primary_label = component_scores[0][0]
        primary_bbox = component_scores[0][4]
        px0, py0, pw, ph = primary_bbox
        px1 = px0 + pw

        # Decide which auxiliary components to retain (e.g. tone marks, upper/lower vowels, multi-part glyphs)
        keep_labels = {primary_label}
        for (lbl, score, area_ratio, touches_border, bbox, centroid) in component_scores[1:]:
            bx, by, bw, bh = bbox
            bx1 = bx + bw
            cx, cy = centroid

            # Horizontal overlap with primary glyph
            horiz_overlap = max(0, min(px1, bx1) - max(px0, bx))
            has_horiz_alignment = horiz_overlap > (0.25 * min(pw, bw))

            # Vertical proximity to primary glyph (above or below, e.g. tone mark or lower vowel)
            is_vertically_aligned = has_horiz_alignment and (
                abs(cy - cy_img) < 0.9 * cy_img or area_ratio > 0.04
            )

            # If it has reasonable size and aligns with the main glyph, keep it
            if area_ratio >= self.min_area_ratio:
                if is_vertically_aligned and not (touches_border and area_ratio < 0.03):
                    keep_labels.add(lbl)
                elif area_ratio > 0.15 and not touches_border:
                    # Large isolated secondary component
                    keep_labels.add(lbl)

        # Build clean binary mask containing only retained components
        clean_mask = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255

        # Create output image: foreground retains bright stroke, background is strictly 0
        if is_dark_text:
            cleaned_np = 255 - np_img
        else:
            cleaned_np = np_img.copy()

        cleaned_np[clean_mask == 0] = 0

        res_img = Image.fromarray(cleaned_np).convert("RGB" if pil_mode == "RGB" else "L")
        if return_mask:
            return res_img, clean_mask
        return res_img


def clean_glyph_artifacts(img: Image.Image) -> Image.Image:
    """Convenience wrapper for focal element artifact cleaning."""
    cleaner = FocalElementCleaner()
    return cleaner.clean(img)
