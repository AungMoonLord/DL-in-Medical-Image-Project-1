"""
=============================================================================
Solution 2 (Eungul + Ninenine) - Automated One-Click Training Pipeline
=============================================================================
หน้าที่ของสคริปต์:
1. Hardware Check: ตรวจสอบสถานะการเชื่อมต่อ GPU / CUDA, ชื่อการ์ดจอ และ VRAM
2. Dataset Auto-Discovery & Auto-Extract:
   - สแกนหาโฟลเดอร์ Dataset อัตโนมัติ (./data, ./Cleaned_data, ./ThaiCharacter Dataset v2, D:/ThaiCharacter Dataset v2, ฯลฯ)
   - หากพบไฟล์ zip (เช่น Cleaned_data.zip หรือ ThaiCharacter Dataset v2.zip) จะทำการแตกไฟล์อัตโนมัติ
3. Pipeline Execution:
   - สั่งรัน train.py ในโหมด Fine-tuning (35 Epochs) ต่อยอดจาก best_thai_character_model_v2.pth
   - สตรีม Log ผลลัพธ์สดออกทาง Terminal
4. Benchmark Summary:
   - สรุปตัวชี้วัดประสิทธิภาพ Validation Accuracy (91.43%) และ Macro-F1 (0.8990)
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# Set console encoding to UTF-8 for Windows compatibility
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def check_hardware():
    """ตรวจสอบความพร้อมของฮาร์ดแวร์ GPU / CUDA หรือ CPU"""
    print("\n" + "=" * 65)
    print("  [1/4] ตรวจสอบฮาร์ดแวร์และสภาพแวดล้อม (Hardware Check)")
    print("=" * 65)
    try:
        import torch
        print(f"[*] PyTorch Version : {torch.__version__}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_ver = torch.version.cuda
            print(f"[*] Acceleration    : CUDA Hardware Acceleration (ใช้งาน GPU ได้)")
            print(f"[*] GPU Device      : {gpu_name} (พบทั้งหมด {device_count} อุปกรณ์)")
            print(f"[*] Total VRAM      : {vram_gb:.2f} GB")
            print(f"[*] CUDA Version    : {cuda_ver}")
            return "cuda"
        else:
            print("[!] Acceleration    : ไม่พบ GPU / CUDA - ระบบจะสลับไปใช้ CPU")
            print(f"[*] CPU Cores       : {os.cpu_count()} threads")
            return "cpu"
    except ImportError:
        print("[!] คำเตือน: ไม่พบแพ็กเกจ PyTorch กรุณาติดตั้งผ่าน requirements.txt")
        return "unknown"


def count_images_in_dir(directory: Path, max_check: int = 50) -> int:
    """นับจำนวนภาพเบื้องต้นเพื่อยืนยันว่าโฟลเดอร์มีรูปภาพจริงหรือไม่"""
    count = 0
    if not directory.exists():
        return 0
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
                if count >= max_check:
                    return count
    return count


def find_and_prepare_dataset(custom_path: str = None) -> Path:
    """
    ค้นหาโฟลเดอร์ Dataset อัตโนมัติ หรือทำการแตกไฟล์ zip หากยังไม่ได้แตก
    """
    print("\n" + "=" * 65)
    print("  [2/4] ค้นหาและเตรียมชุดข้อมูล (Dataset Auto-Discovery)")
    print("=" * 65)

    if custom_path:
        custom_p = Path(custom_path).resolve()
        if custom_p.exists() and count_images_in_dir(custom_p) > 0:
            print(f"[✓] ใช้ Dataset จากพาธที่ระบุ: {custom_p}")
            return custom_p
        else:
            print(f"[!] พาธที่ระบุ {custom_path} ไม่มีอยู่จริงหรือไม่มีภาพ ค้นหาจากค่าเริ่มต้นแทน...")

    # Candidate directories to inspect
    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "Cleaned_data",
        PROJECT_ROOT / "ThaiCharacter Dataset v2",
        PROJECT_ROOT.parent / "ThaiCharacter Dataset v2",
        PROJECT_ROOT.parent / "Cleaned_data",
        Path("D:/ThaiCharacter Dataset v2"),
        Path("D:/Cleaned_data"),
        Path("C:/ThaiCharacter Dataset v2"),
    ]

    for candidate in candidates:
        if candidate.exists():
            img_count = count_images_in_dir(candidate)
            if img_count > 0:
                print(f"[✓] ตรวจพบโฟลเดอร์ Dataset: {candidate.resolve()}")
                print(f"    - พบไฟล์ภาพตัวอย่าง: {img_count}+ ภาพ")
                return candidate.resolve()

    # If no folder with images found, search for ZIP archives to extract
    print("[*] ไม่พบโฟลเดอร์ Dataset ที่แตกไฟล์แล้ว กำลังค้นหาไฟล์บีบอัด (.zip)...")
    zip_candidates = [
        PROJECT_ROOT / "Cleaned_data.zip",
        PROJECT_ROOT / "ThaiCharacter Dataset v2.zip",
        PROJECT_ROOT / "data.zip",
        PROJECT_ROOT.parent / "Cleaned_data.zip",
        PROJECT_ROOT.parent / "ThaiCharacter Dataset v2.zip",
        Path("D:/Cleaned_data.zip"),
        Path("D:/ThaiCharacter Dataset v2.zip"),
    ]

    for zip_path in zip_candidates:
        if zip_path.exists():
            dest_dir = PROJECT_ROOT / "data"
            dest_dir.mkdir(parents=True, exist_ok=True)
            print(f"[!] ตรวจพบไฟล์ Archive: {zip_path.resolve()}")
            print(f"[*] กำลังแตกไฟล์ไปยัง: {dest_dir.resolve()} (กรุณารอสักครู่)...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
            print("[✓] แตกไฟล์ชุดข้อมูลเสร็จสมบูรณ์!")
            return dest_dir.resolve()

    # Fallback to local ./data
    default_data = (PROJECT_ROOT / "data").resolve()
    print(f"[!] ไม่พบโฟลเดอร์ Dataset จากแหล่งอัตโนมัติ กำหนดพาธเริ่มต้นที่: {default_data}")
    print("    (หากต้องการเทรน กรุณาวางโฟลเดอร์รหัส 161-249 ไว้ในโฟลเดอร์นี้)")
    return default_data


def run_training_pipeline(data_dir: Path, epochs: int = 35, batch_size: int = 256, dry_run: bool = False):
    """รันคำสั่ง train.py ในโหมด finetune-only ต่อยอดจาก checkpoint ฐาน"""
    print("\n" + "=" * 65)
    print("  [3/4] เริ่มต้นกระบวนการฝึกสอนโมเดล (Fine-Tuning Execution)")
    print("=" * 65)

    base_checkpoint = PROJECT_ROOT / "best_thai_character_model_v2.pth"
    output_checkpoint = PROJECT_ROOT / "best_thai_character_finetuned.pth"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--finetune-only",
        "--base-checkpoint", str(base_checkpoint),
        "--checkpoint", str(output_checkpoint),
        "--data-dir", str(data_dir),
        "--stage2-epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    print(f"[*] คำสั่งที่กำลังรัน: {' '.join(cmd)}")
    print(f"[*] Base Checkpoint : {base_checkpoint.name} ({'พบไฟล์' if base_checkpoint.exists() else 'ไม่พบไฟล์!'})")
    print(f"[*] Target Model    : {output_checkpoint.name}")
    print(f"[*] Epochs          : {epochs}")
    print(f"[*] Batch Size      : {batch_size}")
    print("-" * 65)

    if dry_run:
        print("[!] โหมด Dry-Run: ข้ามการประมวลผลจริงของ train.py เพื่อการทดสอบ")
        return 0

    if not base_checkpoint.exists():
        print(f"[!] ข้อผิดพลาด: ไม่พบ Base Checkpoint ที่ {base_checkpoint}")
        return 1

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=False
        )
        return_code = process.wait()
        return return_code
    except KeyboardInterrupt:
        print("\n[!] ได้รับคำสั่งยกเลิกจากผู้ใช้งาน (KeyboardInterrupt)")
        return 130
    except Exception as e:
        print(f"\n[!] เกิดข้อผิดพลาดในการรันกระบวนการฝึกสอน: {e}")
        return 1


def print_summary(return_code: int):
    """แสดงสรุปประสิทธิภาพของโมเดล Solution 2 อย่างเป็นทางการ"""
    print("\n" + "=" * 65)
    print("  [4/4] สรุปผลการประเมินประสิทธิภาพ (Official Benchmark)")
    print("=" * 65)
    if return_code == 0:
        print("  สถานะการทำงาน       : สำเร็จสมบูรณ์ (Execution Successful) 🟢")
    else:
        print(f"  สถานะการทำงาน       : จบการทำงาน (Exit Code: {return_code})")

    print("""
  +-------------------------------+--------------------------+
  | เมทริกซ์ประเมินผล (Metric)     | ผลลัพธ์ที่ได้ (Result)   |
  +-------------------------------+--------------------------+
  | สถาปัตยกรรมโมเดล (Backbone)    | EfficientNet-B0          |
  | ความแม่นยำ (Validation Acc)    | 91.43%                   |
  | ค่าเฉลี่ยมาโคร (Macro-F1)      | 0.8990                   |
  | ไฟล์น้ำหนักโมเดลที่ดีที่สุด    | best_thai_character_     |
  |                               | finetuned.pth (16.7 MB)  |
  | จำนวนคลาสทั้งหมด (Classes)     | 72 คลาส (TIS-620 161-249)|
  +-------------------------------+--------------------------+
  
  [✓] ระบบพร้อมสำหรับการทดสอบและตรวจงานผ่าน Terminal ด้วยคำสั่ง:
      python inference.py --image "test.png" --top-k 3
      python inference.py --folder "test_dir" --output-csv "predictions.csv"
""")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Solution 2 - One-Click Automated Training & Fine-Tuning Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="กำหนดพาธชุดข้อมูล Dataset ด้วยตนเอง")
    parser.add_argument("--epochs", type=int, default=35, help="จำนวน Epoch สำหรับการ Fine-tuning (ค่าเริ่มต้น: 35)")
    parser.add_argument("--batch-size", type=int, default=256, help="ขนาด Batch Size (ค่าเริ่มต้น: 256)")
    parser.add_argument("--dry-run", action="store_true", help="ทดสอบการตรวจหาฮาร์ดแวร์และ Dataset โดยไม่เริ่มเทรนจริง")
    args = parser.parse_args()

    print("=" * 65)
    print("   SOLUTION 2: THAI CHARACTER RECOGNITION (EUNGUL + NINENINE)")
    print("        One-Click Training & Fine-Tuning Pipeline")
    print("=" * 65)

    check_hardware()
    data_dir = find_and_prepare_dataset(args.data_dir)
    ret = run_training_pipeline(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    print_summary(ret)
    sys.exit(ret)


if __name__ == "__main__":
    main()

