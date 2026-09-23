import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image
import torch

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset import tis620_to_char, invert_glyph_if_needed, letterbox_pad
from focal_cleaner import FocalElementCleaner
from models import build_model
from prototype_filter import ClassPrototypeEngine
from transforms import PedestalAugmentation, MorphologicalTransform


def test_tis620_mapping():
    assert tis620_to_char(161) == "ก"
    assert tis620_to_char(173) == "ญ"
    assert tis620_to_char(176) == "ฐ"
    assert tis620_to_char(210) == "า"
    assert tis620_to_char(229) == "ๅ"
    print("✓ TIS-620 mapping test passed.")


def test_inversion_and_zero_padding():
    # Create raw dark-on-white image (scanned doc / canvas)
    raw_arr = np.full((30, 20), 255, dtype=np.uint8)
    raw_arr[5:25, 5:15] = 0 # Dark stroke in center
    raw_img = Image.fromarray(raw_arr)

    # Invert first thing: background becomes 0 (black), stroke becomes 255 (bright)
    inv_img = invert_glyph_if_needed(raw_img)
    inv_arr = np.array(inv_img.convert("L"))
    assert inv_arr[0, 0] == 0, "Background must be 0 after inversion"
    assert inv_arr[15, 10] == 255, "Stroke must be 255 after inversion"

    # Letterbox pad with zero-padding
    padded = letterbox_pad(inv_img, target_size=(32, 32), pad_ratio=0.10, fill_color=(0, 0, 0))
    pad_arr = np.array(padded.convert("L"))
    assert padded.size == (32, 32)
    assert pad_arr[0, 0] == 0, "Padded border must be 0 (zero-padding)"
    assert pad_arr[16, 16] > 100, "Centered stroke must be bright"
    print("✓ Inversion and zero-padding test passed.")


def test_focal_cleaner():
    cleaner = FocalElementCleaner()
    # Create white canvas with a main central box and a tiny peripheral border noise box
    img_arr = np.full((50, 50), 255, dtype=np.uint8)
    # Main central stroke
    img_arr[15:35, 15:35] = 0
    # Stray border noise
    img_arr[0:3, 0:3] = 0

    img = Image.fromarray(img_arr)
    cleaned = cleaner.clean(img)
    clean_arr = np.array(cleaned.convert("L"))

    # Stray border noise should be eliminated (0 background)
    assert clean_arr[0, 0] == 0
    # Center stroke should be preserved as bright foreground
    assert clean_arr[20, 20] > 100
    print("✓ Focal element cleaner (0-background) test passed.")


def test_pedestal_aug():
    aug = PedestalAugmentation(p=1.0, bottom_ratio_range=(0.3, 0.3))
    # Operating on inverted image: background is 0, stroke is bright (255)
    img_arr = np.zeros((40, 40), dtype=np.uint8)
    img_arr[5:35, 10:30] = 255 # Bright stroke in center
    img = Image.fromarray(img_arr)

    # For non-target class, should not slice
    out_161 = aug(img, class_number=161)
    assert np.array(out_161)[30, 20] == 255 # Stroke preserved

    # For target class 173 (ญ), should slice bottom 30% to 0
    out_173 = aug(img, class_number=173)
    out_arr = np.array(out_173)
    # Bottom 30% (y >= 28) zero-filled to 0 background
    assert out_arr[32, 20] == 0
    # Upper stroke (y = 10) preserved (255)
    assert out_arr[10, 20] == 255
    print("✓ Pedestal augmentation (0-background) test passed.")


def test_model_builder():
    models = ["custom_cnn", "resnet18", "mobilenet_v3", "efficientnet_b0"]
    for m_name in models:
        m = build_model(m_name, num_classes=72)
        res = 32 if m_name == "custom_cnn" else (224 if m_name == "efficientnet_b0" else 64)
        dummy_x = torch.randn(2, 3, res, res)
        out = m(dummy_x)
        assert out.shape == (2, 72)
    print("✓ Model builder and forward pass tests passed.")


if __name__ == "__main__":
    test_tis620_mapping()
    test_inversion_and_zero_padding()
    test_focal_cleaner()
    test_pedestal_aug()
    test_model_builder()
    print("\n🎉 ALL UNIT TESTS PASSED SUCCESSFULLY!")
