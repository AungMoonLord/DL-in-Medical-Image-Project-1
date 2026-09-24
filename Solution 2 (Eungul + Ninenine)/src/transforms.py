"""
Data Preprocessing and Augmentation Transforms for Thai Character Recognition.
"""

from PIL import Image, ImageOps
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class PadToSquare:
    """
    ขยายขอบภาพให้เป็นสี่เหลี่ยมจัตุรัสโดยการเติมสี (Default: สีขาว 255)
    โดยรักษาอัตราส่วนดั้งเดิมของตัวอักษรไว้ตรงกลางภาพ
    """
    def __init__(self, fill: int = 255):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_side = max(width, height)
        pad_left = (max_side - width) // 2
        pad_right = max_side - width - pad_left
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top
        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill
        )


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Pipeline การแปลงภาพและ Data Augmentation สำหรับชุดฝึกสอน (Train Set)
    - เติมขอบเป็นจัตุรัสสีขาว
    - ย่อ/ขยายเป็นขนาดเป้าหมาย (224x224)
    - ปรับแต่งภาพสุ่ม (RandomAffine) เอียงไม่เกิน 8 องศา, ขยับไม่เกิน 3%, สเกล 95-105%
    - แปลงเป็น Tensor และ Normalize ตามสถิติ ImageNet
    """
    return transforms.Compose([
        PadToSquare(fill=255),
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(
            degrees=8,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Pipeline การแปลงภาพสำหรับชุดตรวจสอบ (Validation Set)
    ไม่มีการ Augmentation เพื่อประเมินผลอย่างเที่ยงตรง
    """
    return transforms.Compose([
        PadToSquare(fill=255),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Pipeline การแปลงภาพสำหรับการทำนายผล (Inference)
    ต้องใช้ขั้นตอนเดียวกับ Validation ทุกประการ
    """
    return get_val_transforms(image_size=image_size)
