# ระบบจำแนกตัวอักษรไทย (Thai Character Recognition)
โมเดล EfficientNet-B0 สำหรับการฝึกสอน (Training), Fine-tuning (35 Epochs) และการทำนายผล (Inference)

---

## 📁 โครงสร้างโปรเจกต์ (Clean & Focused Layout)

```text
Train/
├── src/                      # โมดูลระบบทั้งหมด
│   ├── __init__.py           
│   ├── config.py             # Hyperparameters, TIS-620 Map, Checkpoint Paths
│   ├── dataset.py            # สแกนข้อมูล, Stratified Split, Train-only Augmentation
│   ├── transforms.py         # PadToSquare ขอบขาว 255, ImageNet Normalization
│   ├── models.py             # EfficientNet-B0 สถาปัตยกรรมตรงกับ Checkpoint 100%
│   ├── losses.py             # Class Weighting และ CrossEntropyLoss
│   ├── metrics.py            # Macro-F1, Accuracy, Top-k, Confusion Matrix
│   ├── trainer.py            # 2-Stage Training Loop, Fine-tuning 35 Epochs
│   └── inference.py          # คลาสหลัก ThaiCharacterPredictor
│
├── train.py                  # สคริปต์หลักสำหรับเทรนและ Fine-tune โมเดล
├── inference.py              # สคริปต์หลักสำหรับทำนายผลภาพ (เดี่ยว / ทั้งโฟลเดอร์)
├── requirements.txt          # รายการไลบรารีที่จำเป็น
├── train_final.ipynb         # (ต้นฉบับเดิม)
└── inference.ipynb           # (ต้นฉบับเดิม)
```

---

## 🚀 คำสั่งใช้งานหลัก (2 สคริปต์หลัก: train.py และ inference.py)

### 1. การเทรน / Fine-tuning โมเดล (`train.py`)

- **Fine-tuning ต่อยอดจาก `best_thai_character_model_v2.pth` สู่ `best_thai_character_finetuned.pth` (35 Epochs)**:
  ```bash
  python train.py --finetune-only --data-dir "D:/ThaiCharacter Dataset v2"
  ```

- **Full 2-Stage Training ตั้งแต่เริ่มต้น**:
  ```bash
  python train.py --data-dir ./data --batch-size 256
  ```

---

### 2. การทำนายผลภาพ (`inference.py`)

- **ทำนายภาพเดี่ยว (Single Image)**:
  ```bash
  python inference.py --image "path/to/image.png" --top-k 3
  ```

  **ตัวอย่างผลลัพธ์ใน Terminal**:
  ```text
  ============================================================
   ผลการทำนาย: sample.png
  ============================================================
   อันดับ   ตัวอักษรไทย     รหัส TIS-620       ค่า Confidence (%)
  ------------------------------------------------------------
   #1     ก              161 (0xA1)         99.45%         
   #2     ถ              182 (0xB6)         0.32%          
   #3     ภ              192 (0xC0)         0.11%          
  ============================================================
  ```

- **ทำนายภาพทั้งโฟลเดอร์ (Batch Folder)**:
  ```bash
  python inference.py --folder "path/to/images_dir" --output-csv "results.csv"
  ```
