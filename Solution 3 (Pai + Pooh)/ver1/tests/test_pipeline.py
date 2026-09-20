"""
Unit Tests for Thai Character Classification Pipeline (ver1).
"""

import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset import build_split_dataframes, get_group_key, letterbox_pad, tis620_to_char
from losses import build_loss_fn
from models import build_model
from trainer import build_optimizer, build_scheduler
from transforms import MorphologicalTransform, get_train_transform, get_val_transform


class TestThaiCharacterPipeline(unittest.TestCase):

    def test_tis620_decoding(self):
        self.assertEqual(tis620_to_char(161), "ก")
        self.assertEqual(tis620_to_char(162), "ข")
        self.assertEqual(tis620_to_char(210), "า")
        self.assertEqual(tis620_to_char(249), "๙")

    def test_group_key_pairing(self):
        key_sg = get_group_key("bc_001sg_3_118.jpg")
        key_tg = get_group_key("bc_001tg_3_118.jpg")
        self.assertEqual(key_sg, key_tg, "Counterpart scans sg and tg must map to identical group key")
        self.assertEqual(key_sg, "bc_001_p3")

    def test_letterbox_pad(self):
        # Create a non-square test image (12x18)
        img = Image.new("RGB", (12, 18), color=(200, 200, 200))
        padded = letterbox_pad(img, target_size=(32, 32))
        self.assertEqual(padded.size, (32, 32))

    def test_morphological_transform(self):
        img = Image.new("RGB", (32, 32), color=(255, 255, 255))
        morph = MorphologicalTransform(p_dilate=1.0, p_erode=0.0)
        res = morph(img)
        self.assertEqual(res.size, (32, 32))

    def test_models_forward_backward(self):
        batch_size = 4
        num_classes = 72
        dummy_x_32 = torch.randn(batch_size, 3, 32, 32)
        dummy_y = torch.tensor([0, 1, 71, 42], dtype=torch.long)

        models_to_test = ["custom_cnn", "resnet18", "mobilenet_v3"]
        criterion = build_loss_fn("class_balanced_focal", gamma=2.0)

        for m_name in models_to_test:
            model = build_model(m_name, num_classes=num_classes, pretrained=False)
            logits = model(dummy_x_32)
            self.assertEqual(logits.shape, (batch_size, num_classes))

            loss = criterion(logits, dummy_y)
            self.assertFalse(torch.isnan(loss))
            self.assertFalse(torch.isinf(loss))

            loss.backward()
            optimizer = build_optimizer(model, base_lr=1e-3)
            optimizer.step()
            optimizer.zero_grad()

    def test_scheduler(self):
        model = build_model("custom_cnn", num_classes=72)
        optimizer = build_optimizer(model, base_lr=1e-3)
        scheduler = build_scheduler(optimizer, scheduler_type="cosine_warmup", total_epochs=10, warmup_epochs=2)
        
        # Test warmup phase
        optimizer.step()
        scheduler.step()
        lr_epoch_1 = optimizer.param_groups[0]["lr"]
        self.assertGreater(lr_epoch_1, 0.0)


if __name__ == "__main__":
    unittest.main()
