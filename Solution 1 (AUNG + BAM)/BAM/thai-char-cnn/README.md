# Thai Character & Number Recognition (72 classes) — Full Pipeline

CNN + Transfer Learning + Data Augmentation ตามโจทย์ Project 1 โครงสร้างไฟล์ยึดตาม Chapter 10
(`Net.py` = โครงสร้างโมเดล, `TrainingCNN.py` = ฝึกสอน, `TestingCNN.py` = ทดสอบ, `model.pt` = โมเดลที่เซฟ)

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `data.py` | สแกน dataset, แปลงรหัสโฟลเดอร์ (TIS-620) เป็นตัวอักษร, ตรวจไฟล์เสีย/ภาพซ้ำ, แบ่ง 80/20 แบบไม่รั่ว, Dataset, sampler |
| `augment.py` | PadResize, Data Augmentation, Image Occlusion, Progressive + Adaptive schedule, Sample-aware, perturbation สำหรับทดสอบ robustness |
| `Net.py` | `ThaiCharNet` (pretrained backbone + SE attention + Dropout + 2 heads), EMA |
| `metrics.py` | accuracy / macro-F1 / confusion, ปรับ prior สำหรับ class imbalance |
| `TrainingCNN.py` | Train + Validate + วิเคราะห์ผลหลังเทรน |
| `TestingCNN.py` | ทดสอบกับภาพใหม่ (รองรับ TTA, ensemble, โหมดมีเฉลย) |
| `analyze_data.py` | สร้างตาราง/กราฟอธิบาย dataset สำหรับสไลด์ |
| `make_diagrams.py` | สร้าง `diagrams/architecture.png`, `diagrams/pipeline.png` |

## วิธีรัน

```bash
pip install -r requirements.txt

# 1) อธิบาย dataset (ตาราง จำนวนภาพต่อคลาส กราฟ ตัวอย่างภาพ ตัวอย่าง augmentation)
python analyze_data.py --data /path/to/dataset --out reports

# 2) เทรน + validate (แบ่ง 80:20 อัตโนมัติ)
python TrainingCNN.py --data /path/to/dataset --out runs/r50 --arch resnet50 --img 128 --epochs 40

# 3) ทดสอบกับข้อมูลใหม่
python TestingCNN.py --ckpt runs/r50/model.pt --input test_images/ --tta --out predictions.csv
python TestingCNN.py --ckpt runs/r50/model.pt --input test_dir/ --labeled --tta      # ถ้าจัดเป็นโฟลเดอร์คลาส

# 4) (ทางเลือก) โมเดลสุดท้ายที่ใช้ข้อมูล 100% หลังเลือกค่าต่างๆ ได้แล้ว
python TrainingCNN.py --data /path/to/dataset --out runs/final --arch resnet50 --epochs 40 --full --alpha 0.5
```
`--data` ชี้ไปที่โฟลเดอร์ที่มี `round2/` หรือโฟลเดอร์คลาส (161, 162, ...) โดยตรงก็ได้
บน Windows ถ้ามีปัญหา multiprocessing ให้เติม `--workers 0`

ตัวเลือกที่ควรลองเทียบ: `--arch` (resnet50 / efficientnet_b0 / convnext_tiny / mobilenet_v3_large), `--img 96/128/160`,
`--fine_detail`, `--sampler_power 0/0.5/1`, `--group_w 0` (ปิด multi-task), `--no_occlusion`, `--no_adaptive_aug`, `--no_attention`

## ตรงกับเกณฑ์ที่อาจารย์กำหนด

| เกณฑ์ | ส่วนที่ตอบ |
|---|---|
| ทำนายภาพทดสอบ (5%) | `TestingCNN.py` → `predictions.csv` |
| อันดับประสิทธิภาพ (3%) | Transfer learning + augmentation + จัดการ imbalance + TTA/ensemble |
| Transfer Learning (1.5%) | `Net.py`: backbone ImageNet-pretrained จาก `torchvision.models` fine-tune ทั้งโมเดล (backbone LR = 0.1 × head) |
| Data Augmentation (1.5%) | `augment.py`: geometric, photometric, stroke-width, Image Occlusion (Random Erase / Cutout / Hide-and-Seek) |
| เทคนิค/แนวคิดที่น่าสนใจ (2%) | Sample-aware + Progressive + Adaptive augmentation, SE attention, Multi-task (glyph group), logit adjustment (α), split กันภาพซ้ำรั่ว |
| อธิบายชุดข้อมูล | `reports/class_counts.csv`, `class_distribution.png`, `sample_grid.png` |
| วิเคราะห์ความท้าทาย | `reports/data_report.json` (imbalance, ภาพซ้ำ, polarity, ขนาดภาพ) + `top_confusions.txt` |
| โครงสร้าง CNN / การทำงาน | `diagrams/architecture.png` |
| ขั้นตอนการฝึกสอน | `diagrams/pipeline.png`, `curves.png` |
| Accuracy Rate | `runs/*/final_metrics.json`, `history.csv` |
| แบ่ง Train:Validation 80:20 | `split.json` (แบ่งต่อคลาส, ภาพซ้ำอยู่ฝั่งเดียวกัน) |
| แนะนำสมาชิกกลุ่ม | ใส่เองในสไลด์ |

## หลักคิดของเทคนิคสำคัญ

**Imbalance (ต่างกัน 5,025 เท่า):** (1) sampler แบบ sqrt-inverse-frequency (2) Sample-aware augmentation: คลาสหายากถูก augment แรงกว่า
(3) Logit adjustment หลังเทรน: ตัวแปร `alpha` ตัวเดียว (0 = balanced, 1 = ตามธรรมชาติ) เลือกจาก val โดยไม่ต้องเทรนใหม่ ดูตาราง `alpha_sweep.json`

**ตัวอักษรคล้ายกัน:** ใช้ภาพละเอียด + ไม่ยืดภาพ, SE attention, `--fine_detail`, multi-task head (ตัวอักษรหลัก + กลุ่มรูปร่าง),
TTA/ensemble, และ `errors/` + `top_confusions.txt` ไว้ตรวจว่าคู่ไหนสับสน/ป้ายผิด

**Augmentation ที่ปลอดภัยกับอักษรไทย:** ไม่ใช้ flip, CutMix, MixUp (ทำลายสระ/วรรณยุกต์ที่เล็กมาก) สระ-วรรณยุกต์ใช้ geometry เบาและไม่ occlusion,
มีตัวกันภาพจางจนมองไม่เห็น, ดูผลจริงได้ที่ `reports/augmentation_preview.png`

**ปัญหาของ image data ที่ป้องกัน:** ไฟล์เสีย, PNG โปร่งใส (พื้นหลังโปร่งใสกลายเป็นดำ), EXIF หมุน, 16-bit/palette, ภาพซ้ำข้าม train/val (dHash),
ภาพซ้ำข้ามคลาส (ป้ายผิด), ขั้วสีกลับ (ตัวขาวพื้นดำ), aspect ratio, preprocessing ต่างกันระหว่าง train/test (ใช้โค้ดเดียวกัน),
seed/split ถูกเซฟไว้ทำซ้ำได้

## ข้อจำกัด (ซื่อสัตย์)

- โค้ดทดสอบกับข้อมูลจำลองบน CPU เท่านั้น (pipeline รันครบ และเรียนรู้ชุดง่ายได้ 100%) **ยังไม่เคยรันกับข้อมูลจริงของอาจารย์** และยังไม่ได้ทดสอบการโหลดน้ำหนัก pretrained
  (ต้องต่ออินเทอร์เน็ตครั้งแรก) ค่า accuracy จริงยังไม่ทราบ
- `alpha` และตัวเลข val ควรตีความอย่างระวัง: คลาสที่มี 1-4 ภาพ (ฃ ฑ ฬ ๗) วัดผลบน val ไม่ได้
- ถ้าภาพใน dataset มาจากคนเขียน/แหล่งเดียวกัน val จะดีกว่า unseen จริง ให้ดูตาราง robustness ใน `final_metrics.json` ประกอบ
- ค่า default (epochs, LR, กำลัง augmentation) เป็นจุดเริ่มต้นที่สมเหตุสมผล ยังไม่ได้จูนกับข้อมูลจริง
