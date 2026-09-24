"""
Universal Inference Engine and Dynamic Checkpoint Loader (ver2).

Features:
1. Universal Model Loader: Auto-detects architecture from checkpoint state_dict
   (CustomGlyphCNN, AdaptedResNet18, AdaptedMobileNetV3, Solution 2 EfficientNet-B0, Solution 1 checkpoints).
2. Adaptive Image Preprocessor: Bounding-box detection, Center-of-Mass alignment,
   configurable safety padding, and model-specific resolution scaling.
3. Top-K confidence extraction with comprehensive Thai character metadata.
4. Real-time base64 letterbox tensor visualization.
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

try:
    from .dataset import build_split_dataframes, invert_glyph_if_needed, letterbox_pad, tis620_to_char
    from .focal_cleaner import FocalElementCleaner
    from .models import AdaptedEfficientNetB0, AdaptedMobileNetV3, AdaptedResNet18, CustomGlyphCNN, build_model
    from .transforms import get_val_transform
except (ImportError, ValueError):
    from dataset import build_split_dataframes, invert_glyph_if_needed, letterbox_pad, tis620_to_char
    from focal_cleaner import FocalElementCleaner
    from models import AdaptedEfficientNetB0, AdaptedMobileNetV3, AdaptedResNet18, CustomGlyphCNN, build_model
    from transforms import get_val_transform

THAI_CHAR_METADATA: Dict[int, Dict[str, str]] = {
    161: {"char": "ก", "name_th": "ก ไก่", "name_en": "Ko Kai", "type": "Consonant (Middle)", "meaning": "Chicken"},
    162: {"char": "ข", "name_th": "ข ไข่", "name_en": "Kho Khai", "type": "Consonant (High)", "meaning": "Egg"},
    163: {"char": "ฃ", "name_th": "ฃ ขวด", "name_en": "Kho Khuat", "type": "Consonant (Obsolete)", "meaning": "Bottle"},
    164: {"char": "ค", "name_th": "ค ควาย", "name_en": "Kho Khwai", "type": "Consonant (Low)", "meaning": "Water Buffalo"},
    167: {"char": "ง", "name_th": "ง งู", "name_en": "Ngo Ngu", "type": "Consonant (Low)", "meaning": "Snake"},
    168: {"char": "จ", "name_th": "จ จาน", "name_en": "Cho Chan", "type": "Consonant (Middle)", "meaning": "Plate"},
    169: {"char": "ฉ", "name_th": "ฉ ฉิ่ง", "name_en": "Cho Ching", "type": "Consonant (High)", "meaning": "Cymbals"},
    170: {"char": "ช", "name_th": "ช ช้าง", "name_en": "Cho Chang", "type": "Consonant (Low)", "meaning": "Elephant"},
    171: {"char": "ซ", "name_th": "ซ โซ่", "name_en": "So So", "type": "Consonant (Low)", "meaning": "Chain"},
    173: {"char": "ญ", "name_th": "ญ หญิง", "name_en": "Yo Ying", "type": "Consonant (Low)", "meaning": "Woman"},
    175: {"char": "ฏ", "name_th": "ฏ ปฏัก", "name_en": "To Patak", "type": "Consonant (Middle)", "meaning": "Goad / Spear"},
    176: {"char": "ฐ", "name_th": "ฐ ฐาน", "name_en": "Tho Than", "type": "Consonant (High)", "meaning": "Pedestal"},
    177: {"char": "ฑ", "name_th": "ฑ มณโฑ", "name_en": "Tho Montho", "type": "Consonant (Low)", "meaning": "Montho (Queen)"},
    178: {"char": "ฒ", "name_th": "ฒ ผู้เฒ่า", "name_en": "Tho Phuthao", "type": "Consonant (Low)", "meaning": "Elderly Person"},
    179: {"char": "ณ", "name_th": "ณ เณร", "name_en": "No Nen", "type": "Consonant (Low)", "meaning": "Novice Monk"},
    180: {"char": "ด", "name_th": "ด เด็ก", "name_en": "Do Dek", "type": "Consonant (Middle)", "meaning": "Child"},
    181: {"char": "ต", "name_th": "ต เต่า", "name_en": "To Tao", "type": "Consonant (Middle)", "meaning": "Turtle"},
    182: {"char": "ถ", "name_th": "ถ ถุง", "name_en": "Tho Thung", "type": "Consonant (High)", "meaning": "Bag / Sack"},
    183: {"char": "ท", "name_th": "ท ทหาร", "name_en": "Tho Thahan", "type": "Consonant (Low)", "meaning": "Soldier"},
    184: {"char": "ธ", "name_th": "ธ ธง", "name_en": "Tho Thong", "type": "Consonant (Low)", "meaning": "Flag"},
    185: {"char": "น", "name_th": "น หนู", "name_en": "No Nu", "type": "Consonant (Low)", "meaning": "Mouse"},
    186: {"char": "บ", "name_th": "บ ใบไม้", "name_en": "Bo Baimai", "type": "Consonant (Middle)", "meaning": "Leaf"},
    187: {"char": "ป", "name_th": "ป ปลา", "name_en": "Po Pla", "type": "Consonant (Middle)", "meaning": "Fish"},
    188: {"char": "ผ", "name_th": "ผ ผึ้ง", "name_en": "Pho Phueng", "type": "Consonant (High)", "meaning": "Bee"},
    189: {"char": "ฝ", "name_th": "ฝ ฝา", "name_en": "Fo Fa", "type": "Consonant (High)", "meaning": "Lid / Cover"},
    190: {"char": "พ", "name_th": "พ พาน", "name_en": "Pho Phan", "type": "Consonant (Low)", "meaning": "Tray on Pedestal"},
    191: {"char": "ฟ", "name_th": "ฟ ฟัน", "name_en": "Fo Fan", "type": "Consonant (Low)", "meaning": "Tooth"},
    192: {"char": "ภ", "name_th": "ภ สำเภา", "name_en": "Pho Samphao", "type": "Consonant (Low)", "meaning": "Sailboat"},
    193: {"char": "ม", "name_th": "ม ม้า", "name_en": "Mo Ma", "type": "Consonant (Low)", "meaning": "Horse"},
    194: {"char": "ย", "name_th": "ย ยักษ์", "name_en": "Yo Yak", "type": "Consonant (Low)", "meaning": "Giant"},
    195: {"char": "ร", "name_th": "ร เรือ", "name_en": "Ro Ruea", "type": "Consonant (Low)", "meaning": "Boat"},
    196: {"char": "ฤ", "name_th": "ตัว ฤ", "name_en": "Rue", "type": "Vocalic Consonant", "meaning": "Sanskrit Vocalic R"},
    197: {"char": "ล", "name_th": "ล ลิง", "name_en": "Lo Ling", "type": "Consonant (Low)", "meaning": "Monkey"},
    199: {"char": "ว", "name_th": "ว แหวน", "name_en": "Wo Waen", "type": "Consonant (Low)", "meaning": "Ring"},
    200: {"char": "ศ", "name_th": "ศ ศาลา", "name_en": "So Sala", "type": "Consonant (High)", "meaning": "Pavilion"},
    201: {"char": "ษ", "name_th": "ษ ฤๅษี", "name_en": "So Rusi", "type": "Consonant (High)", "meaning": "Hermit"},
    202: {"char": "ส", "name_th": "ส เสือ", "name_en": "So Suea", "type": "Consonant (High)", "meaning": "Tiger"},
    203: {"char": "ห", "name_th": "ห หีบ", "name_en": "Ho Hip", "type": "Consonant (High)", "meaning": "Chest / Box"},
    204: {"char": "ฬ", "name_th": "ฬ จุฬา", "name_en": "Lo Chula", "type": "Consonant (Low)", "meaning": "Kite"},
    205: {"char": "อ", "name_th": "อ อ่าง", "name_en": "O Ang", "type": "Consonant (Middle)", "meaning": "Basin"},
    206: {"char": "ฮ", "name_th": "ฮ นกฮูก", "name_en": "Ho Nokhuk", "type": "Consonant (Low)", "meaning": "Owl"},
    207: {"char": "ฯ", "name_th": "ไปยาลน้อย", "name_en": "Paiyannoi", "type": "Punctuation", "meaning": "Ellipsis / Abbreviation"},
    209: {"char": "ั", "name_th": "ไม้หันอากาศ", "name_en": "Mai Han-Akat", "type": "Upper Vowel", "meaning": "Short 'a' Vowel"},
    210: {"char": "า", "name_th": "สระ อา", "name_en": "Sara Aa", "type": "Following Vowel", "meaning": "Long 'aa' Vowel"},
    212: {"char": "ิ", "name_th": "สระ อิ", "name_en": "Sara I", "type": "Upper Vowel", "meaning": "Short 'i' Vowel"},
    213: {"char": "ี", "name_th": "สระ อี", "name_en": "Sara Ii", "type": "Upper Vowel", "meaning": "Long 'ee' Vowel"},
    214: {"char": "ึ", "name_th": "สระ อึ", "name_en": "Sara Ue", "type": "Upper Vowel", "meaning": "Short 'ue' Vowel"},
    215: {"char": "ื", "name_th": "สระ อือ", "name_en": "Sara Uee", "type": "Upper Vowel", "meaning": "Long 'uee' Vowel"},
    216: {"char": "ุ", "name_th": "สระ อุ", "name_en": "Sara U", "type": "Lower Vowel", "meaning": "Short 'u' Vowel"},
    217: {"char": "ู", "name_th": "สระ อู", "name_en": "Sara Uu", "type": "Lower Vowel", "meaning": "Long 'oo' Vowel"},
    224: {"char": "เ", "name_th": "สระ เอ", "name_en": "Sara E", "type": "Leading Vowel", "meaning": "Long 'e' Vowel"},
    225: {"char": "แ", "name_th": "สระ แอ", "name_en": "Sara Ae", "type": "Leading Vowel", "meaning": "Long 'ae' Vowel"},
    226: {"char": "โ", "name_th": "สระ โอ", "name_en": "Sara O", "type": "Leading Vowel", "meaning": "Long 'o' Vowel"},
    227: {"char": "ใ", "name_th": "สระ ไอไม้ม้วน", "name_en": "Sara Ai Maimuan", "type": "Leading Vowel", "meaning": "'ai' Vowel (Rolled)"},
    228: {"char": "ไ", "name_th": "สระ ไอไม้มลาย", "name_en": "Sara Ai Maimalai", "type": "Leading Vowel", "meaning": "'ai' Vowel"},
    229: {"char": "ๅ", "name_th": "ลากข้างยาว", "name_en": "Lakkhangyao", "type": "Vowel Length Sign", "meaning": "Vowel elongation"},
    230: {"char": "ๆ", "name_th": "ไม้ยมก", "name_en": "Maiyamok", "type": "Punctuation", "meaning": "Repetition Mark"},
    231: {"char": "็", "name_th": "ไม้ไต่คู้", "name_en": "Maitaikhu", "type": "Tone / Vowel Sign", "meaning": "Vowel shortener"},
    232: {"char": "่", "name_th": "ไม้เอก", "name_en": "Mai Ek", "type": "Tone Marker (1st)", "meaning": "Low Tone Mark"},
    233: {"char": "้", "name_th": "ไม้โท", "name_en": "Mai Tho", "type": "Tone Marker (2nd)", "meaning": "Falling Tone Mark"},
    234: {"char": "๊", "name_th": "ไม้ตรี", "name_en": "Mai Tri", "type": "Tone Marker (3rd)", "meaning": "High Tone Mark"},
    236: {"char": "์", "name_th": "ไม้ทัณฑฆาต (การันต์)", "name_en": "Thanthakhat", "type": "Tone / Silence Mark", "meaning": "Cancels sound of letter"},
    240: {"char": "๐", "name_th": "เลขศูนย์ไทย", "name_en": "Sun (0)", "type": "Thai Numeral", "meaning": "Digit 0"},
    241: {"char": "๑", "name_th": "เลขหนึ่งไทย", "name_en": "Nueng (1)", "type": "Thai Numeral", "meaning": "Digit 1"},
    242: {"char": "๒", "name_th": "เลขสองไทย", "name_en": "Song (2)", "type": "Thai Numeral", "meaning": "Digit 2"},
    243: {"char": "๓", "name_th": "เลขสามไทย", "name_en": "Sam (3)", "type": "Thai Numeral", "meaning": "Digit 3"},
    244: {"char": "๔", "name_th": "เลขสี่ไทย", "name_en": "Si (4)", "type": "Thai Numeral", "meaning": "Digit 4"},
    245: {"char": "๕", "name_th": "เลขห้าไทย", "name_en": "Ha (5)", "type": "Thai Numeral", "meaning": "Digit 5"},
    246: {"char": "๖", "name_th": "เลขหกไทย", "name_en": "Hok (6)", "type": "Thai Numeral", "meaning": "Digit 6"},
    247: {"char": "๗", "name_th": "เลขเจ็ดไทย", "name_en": "Chet (7)", "type": "Thai Numeral", "meaning": "Digit 7"},
    248: {"char": "๘", "name_th": "เลขแปดไทย", "name_en": "Paet (8)", "type": "Thai Numeral", "meaning": "Digit 8"},
    249: {"char": "๙", "name_th": "เลขเก้าไทย", "name_en": "Kao (9)", "type": "Thai Numeral", "meaning": "Digit 9"},
}


class UniversalModelLoader:
    """
    Introspects checkpoint weights to dynamically construct the matching architecture
    and resolution with zero configuration mismatch.
    """

    @staticmethod
    def inspect_and_load(
        checkpoint_path: Union[str, Path],
        device: torch.device,
        num_classes: int = 72,
    ) -> Tuple[nn.Module, str, int, Optional[Dict[int, int]]]:
        """
        Loads checkpoint and detects architecture:
        Returns: (model, model_type, target_resolution, custom_class_to_idx)
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        try:
            loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            loaded = torch.load(ckpt_path, map_location=device)

        state_dict = loaded
        custom_class_map = None
        target_res = None
        detected_num_classes = num_classes
        arch_hint = None

        if isinstance(loaded, dict):
            arch_hint = loaded.get("model_arch") or loaded.get("model_name") or loaded.get("arch")
            if arch_hint:
                arch_hint = str(arch_hint).lower().strip()

            if "class_to_idx" in loaded and isinstance(loaded["class_to_idx"], dict):
                custom_class_map = {int(k) if str(k).isdigit() else k: int(v) for k, v in loaded["class_to_idx"].items()}
            elif "idx_to_class" in loaded and isinstance(loaded["idx_to_class"], dict):
                custom_class_map = {int(v) if str(v).isdigit() else v: int(k) for k, v in loaded["idx_to_class"].items()}

            if "num_classes" in loaded:
                detected_num_classes = int(loaded["num_classes"])
            elif custom_class_map:
                detected_num_classes = len(custom_class_map)

            if "image_size" in loaded:
                target_res = int(loaded["image_size"])
            elif "target_res" in loaded:
                target_res = int(loaded["target_res"])

            if "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]

        # Clean state dict prefixes if wrapped in DDP / module
        clean_sd = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "").replace("model.", "")
            clean_sd[clean_k] = v

        keys = list(clean_sd.keys())

        # Auto-detect num_classes from output head if possible
        if "classifier.6.weight" in clean_sd:
            detected_num_classes = clean_sd["classifier.6.weight"].shape[0]
        elif "classifier.4.weight" in clean_sd:
            detected_num_classes = clean_sd["classifier.4.weight"].shape[0]
        elif "classifier.1.weight" in clean_sd:
            detected_num_classes = clean_sd["classifier.1.weight"].shape[0]

        # Architecture detection heuristics
        if arch_hint in ("custom_cnn", "customglyphcnn", "glyph_cnn") or any("stage1" in k for k in keys):
            model_type = "custom_cnn"
            model = CustomGlyphCNN(num_classes=detected_num_classes)
            target_res = target_res or 32

        elif arch_hint in ("resnet18", "resnet_18", "adapted_resnet18") or \
             any("features.4.0.conv1" in k for k in keys) or \
             any("layer1" in k for k in keys) or \
             ("classifier.2.weight" in clean_sd and clean_sd["classifier.2.weight"].shape[1] == 512):
            model_type = "resnet18"
            stem_k = "features.0.weight" if "features.0.weight" in clean_sd else ("conv1.weight" if "conv1.weight" in clean_sd else None)
            adapt_stem = (clean_sd[stem_k].shape[-1] == 3) if stem_k and stem_k in clean_sd else False
            model = AdaptedResNet18(num_classes=detected_num_classes, pretrained=False, adapt_stem=adapt_stem)
            target_res = target_res or 64

        elif arch_hint in ("efficientnet_b0", "efficientnet", "effnet_b0", "solution2") or \
             any("features.1.0.block" in k for k in keys) or \
             ("classifier.1.weight" in clean_sd and clean_sd["classifier.1.weight"].shape[1] == 1280):
            model_type = "efficientnet_b0"
            model = AdaptedEfficientNetB0(num_classes=detected_num_classes, pretrained=False)
            target_res = target_res or 224

        elif arch_hint in ("mobilenet_v3", "mobilenetv3", "adapted_mobilenet") or \
             any("features.1.block" in k for k in keys) or \
             ("classifier.1.weight" in clean_sd and clean_sd["classifier.1.weight"].shape[0] == 256):
            model_type = "mobilenet_v3"
            model = AdaptedMobileNetV3(num_classes=detected_num_classes, pretrained=False)
            target_res = target_res or 64

        else:
            # Fallback to Custom CNN
            model_type = "custom_cnn"
            model = CustomGlyphCNN(num_classes=detected_num_classes)
            target_res = target_res or 32

        try:
            model.load_state_dict(clean_sd, strict=True)
        except Exception:
            model.load_state_dict(clean_sd, strict=False)

        model = model.to(device).eval()
        return model, model_type, target_res, custom_class_map


class ThaiCharacterInferenceEngine:
    """
    ver2 Unified Inference Engine supporting multi-solution checkpoints,
    configurable letterbox padding, focal artifact cleaning, Top-K predictions,
    and batch evaluation.
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.dataset_dir = dataset_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.val_transform = get_val_transform()
        self.focal_cleaner = FocalElementCleaner()

        # Fast discovery of class mapping without slow recursive disk scanning
        if self.dataset_dir:
            try:
                roots = [Path(p) for p in (self.dataset_dir if isinstance(self.dataset_dir, (list, tuple)) else [self.dataset_dir])]
                classes = set()
                for root in roots:
                    if root.exists():
                        for d in root.iterdir():
                            if d.is_dir() and d.name.isdigit():
                                classes.add(int(d.name))
                if classes:
                    sorted_classes = sorted(list(classes))
                    self.class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
                else:
                    sorted_classes = sorted(list(THAI_CHAR_METADATA.keys()))
                    self.class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
            except Exception:
                sorted_classes = sorted(list(THAI_CHAR_METADATA.keys()))
                self.class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
        else:
            sorted_classes = sorted(list(THAI_CHAR_METADATA.keys()))
            self.class_to_idx = {c: i for i, c in enumerate(sorted_classes)}

        self.idx_to_class = {idx: num for num, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def load_model(
        self,
        model_name: str,
        checkpoint_path: Union[str, Path],
    ) -> nn.Module:
        """Loads and caches a model checkpoint dynamically."""
        model, model_type, target_res, custom_class_map = UniversalModelLoader.inspect_and_load(
            checkpoint_path=checkpoint_path,
            device=self.device,
            num_classes=self.num_classes,
        )

        self.models[model_name] = model
        self.model_configs[model_name] = {
            "type": model_type,
            "target_res": target_res,
            "custom_class_map": custom_class_map,
            "ckpt_path": str(checkpoint_path),
        }
        return model

    def preprocess_image(
        self,
        image_input: Union[Image.Image, np.ndarray, str, bytes],
        target_size: Tuple[int, int] = (32, 32),
        pad_ratio: float = 0.05,
        use_focal_cleaner: bool = False,
        autocrop: bool = True,
    ) -> Tuple[torch.Tensor, Image.Image, Image.Image]:
        """
        Preprocesses various image formats:
        Returns: (tensor, letterboxed_img, cropped_glyph)
        """
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                header, encoded = image_input.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # 1. Invert incoming image first thing (0-background, bright-stroke glyph)
        pil_img = invert_glyph_if_needed(pil_img)

        # 2. Optional focal artifact cleaner (operates on 0-background)
        if use_focal_cleaner:
            pil_img = self.focal_cleaner.clean(pil_img)

        # 3. Autocrop to bounding box of bright foreground content
        if autocrop:
            np_img = np.array(pil_img)
            gray = np.mean(np_img, axis=2) if np_img.ndim == 3 else np_img
            coords = np.argwhere(gray > 30)

            if len(coords) > 10:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1
                # Add safety margin
                pad_px = max(2, int(min(pil_img.width, pil_img.height) * 0.03))
                x0 = max(0, x0 - pad_px)
                y0 = max(0, y0 - pad_px)
                x1 = min(pil_img.width, x1 + pad_px)
                y1 = min(pil_img.height, y1 + pad_px)
                cropped_glyph = pil_img.crop((x0, y0, x1, y1))
            else:
                cropped_glyph = pil_img
        else:
            cropped_glyph = pil_img

        # 4. Letterbox pad to target square size with zero padding (0, 0, 0)
        letterboxed = letterbox_pad(cropped_glyph, target_size=target_size, pad_ratio=pad_ratio, fill_color=(0, 0, 0))
        tensor = self.val_transform(letterboxed).unsqueeze(0).to(self.device)
        return tensor, letterboxed, cropped_glyph

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, str, bytes],
        model_name: str = "custom_cnn",
        top_k: int = 5,
        pad_ratio: float = 0.05,
        use_focal_cleaner: bool = False,
        autocrop: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs single-sample inference and returns structured predictions.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' is not loaded. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        cfg = self.model_configs.get(model_name, {})
        target_res = cfg.get("target_res", 32)
        custom_class_map = cfg.get("custom_class_map")

        tensor, letterboxed, cropped = self.preprocess_image(
            image_input,
            target_size=(target_res, target_res),
            pad_ratio=pad_ratio,
            use_focal_cleaner=use_focal_cleaner,
            autocrop=autocrop,
        )

        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        top_k_indices = np.argsort(probs)[::-1][:top_k]

        # Lookup idx to class
        idx_map = self.idx_to_class
        if custom_class_map is not None:
            idx_map = {int(v): (int(k) if str(k).isdigit() else k) for k, v in custom_class_map.items()}

        predictions: List[Dict[str, Any]] = []
        for idx in top_k_indices:
            class_num = idx_map.get(int(idx), int(idx))
            meta = THAI_CHAR_METADATA.get(class_num if isinstance(class_num, int) else -1, {
                "char": tis620_to_char(class_num) if isinstance(class_num, int) or (isinstance(class_num, str) and class_num.isdigit()) else str(class_num),
                "name_th": f"รหัส {class_num}",
                "name_en": f"Class {class_num}",
                "type": "General",
                "meaning": "-",
            })

            predictions.append({
                "class_number": class_num,
                "class_idx": int(idx),
                "character": meta["char"],
                "name_th": meta["name_th"],
                "name_en": meta["name_en"],
                "type": meta["type"],
                "meaning": meta["meaning"],
                "probability": float(probs[idx]),
                "confidence_pct": round(float(probs[idx]) * 100, 2),
            })

        # Previews in base64
        buf_let = io.BytesIO()
        letterboxed.save(buf_let, format="PNG")
        letterbox_b64 = "data:image/png;base64," + base64.b64encode(buf_let.getvalue()).decode("utf-8")

        buf_crop = io.BytesIO()
        cropped.save(buf_crop, format="PNG")
        crop_b64 = "data:image/png;base64," + base64.b64encode(buf_crop.getvalue()).decode("utf-8")

        return {
            "model_name": model_name,
            "model_type": cfg.get("type", "unknown"),
            "resolution": f"{target_res}x{target_res}",
            "top_prediction": predictions[0] if predictions else None,
            "predictions": predictions,
            "preprocessed_preview": letterbox_b64,
            "crop_preview": crop_b64,
        }

    @torch.no_grad()
    def evaluate_loader(
        self,
        val_loader: torch.utils.data.DataLoader,
        model_name: str = "custom_cnn",
    ) -> Dict[str, Any]:
        """
        Evaluates a loaded model across a DataLoader and returns comprehensive metrics.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' is not loaded. Available: {list(self.models.keys())}")

        import time
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

        model = self.models[model_name]
        cfg = self.model_configs.get(model_name, {})
        target_res = cfg.get("target_res", 32)
        custom_class_map = cfg.get("custom_class_map")

        idx_map = self.idx_to_class
        if custom_class_map is not None:
            idx_map = {int(v): (int(k) if str(k).isdigit() else k) for k, v in custom_class_map.items()}

        all_preds = []
        all_targets = []
        all_probs = []
        top3_correct = 0
        top5_correct = 0
        total = 0

        t0 = time.time()
        for tensors, targets, _ in val_loader:
            tensors = tensors.to(self.device)
            if tensors.size(-1) != target_res:
                tensors = torch.nn.functional.interpolate(tensors, size=(target_res, target_res), mode="bilinear", align_corners=False)

            targets_dev = targets.to(self.device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1).indices
            top3_correct += (top3 == targets_dev.unsqueeze(1)).any(dim=-1).sum().item()

            top5 = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
            top5_correct += (top5 == targets_dev.unsqueeze(1)).any(dim=-1).sum().item()

            total += targets.size(0)

        elapsed = time.time() - t0
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        num_cls = len(idx_map)
        labels_list = list(range(num_cls))
        target_names = [tis620_to_char(idx_map[i]) if isinstance(idx_map[i], int) else str(idx_map[i]) for i in labels_list]

        cm = confusion_matrix(all_targets, all_preds, labels=labels_list)
        top1_acc = float((all_preds == all_targets).mean()) * 100.0
        top3_acc = float(top3_correct / max(1, total)) * 100.0
        top5_acc = float(top5_correct / max(1, total)) * 100.0
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0)) * 100.0
        weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0)) * 100.0
        macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0)) * 100.0
        macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0)) * 100.0

        throughput = round(total / max(0.001, elapsed), 1)

        return {
            "model_name": model_name,
            "model_type": cfg.get("type", "unknown"),
            "resolution": f"{target_res}x{target_res}",
            "total_samples": total,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_samples_per_sec": throughput,
            "top1_accuracy": round(top1_acc, 2),
            "top3_accuracy": round(top3_acc, 2),
            "top5_accuracy": round(top5_acc, 2),
            "macro_f1": round(macro_f1, 2),
            "weighted_f1": round(weighted_f1, 2),
            "macro_precision": round(macro_prec, 2),
            "macro_recall": round(macro_rec, 2),
            "confusion_matrix": cm,
            "target_names": target_names,
            "all_preds": all_preds,
            "all_targets": all_targets,
            "all_probs": all_probs,
        }

