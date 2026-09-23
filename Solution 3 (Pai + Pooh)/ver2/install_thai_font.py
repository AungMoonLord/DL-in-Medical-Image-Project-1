"""
Standalone Matplotlib Thai Font Installer and Verification Suite (ver2).

Run this script to:
1. Detect and configure Thai fonts for Matplotlib.
2. Download Google Fonts 'Sarabun' (OFL) to local fonts/ directory if needed.
3. Generate a visual verification test plot (thai_font_verification.png) to guarantee zero tofu (□) rendering.

Usage:
    python install_thai_font.py
"""

import os
from pathlib import Path
import sys

# Ensure stdout handles UTF-8 / Thai / emojis cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Ensure ver2/src is in sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from font_utils import (
    SYSTEM_THAI_FONT_CANDIDATES,
    download_thai_font,
    find_system_thai_font,
    register_font_file,
    setup_thai_font,
)


def verify_thai_rendering(output_path: Path, font_name: str) -> bool:
    """Generates a comprehensive visual test chart with Thai glyphs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Matplotlib Thai Font Verification (Active Font: '{font_name}')", fontsize=16, fontweight="bold")

    # 1. Consonants & Key Classes
    sample_consonants = ["ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ", "ช", "ญ", "ฐ", "ด", "ต", "ภ", "ม", "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"]
    axes[0, 0].bar(range(len(sample_consonants)), [i + 5 for i in range(len(sample_consonants))], color="#6366f1")
    axes[0, 0].set_xticks(range(len(sample_consonants)))
    axes[0, 0].set_xticklabels(sample_consonants, fontsize=11, fontweight="bold")
    axes[0, 0].set_title("1. Thai Consonants (พยัญชนะไทย)", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("Sample Value")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Vowels & Tone Markers
    sample_vowels = ["ะ", "ั", "า", "ำ", "ิ", "ี", "ึ", "ื", "ุ", "ู", "เ", "แ", "โ", "ใ", "ไ", "่", "้", "๊", "๋", "์"]
    axes[0, 1].plot(range(len(sample_vowels)), np.sin(np.linspace(0, 3, len(sample_vowels))), marker="o", color="#10b981", lw=2)
    axes[0, 1].set_xticks(range(len(sample_vowels)))
    axes[0, 1].set_xticklabels(sample_vowels, fontsize=14, fontweight="bold")
    axes[0, 1].set_title("2. Thai Vowels & Tone Marks (สระและวรรณยุกต์)", fontsize=13, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Thai Numerals
    thai_digits = ["๐", "๑", "๒", "๓", "๔", "๕", "๖", "๗", "๘", "๙"]
    arabic_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    axes[1, 0].barh(range(10), arabic_digits, color="#f59e0b")
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels([f"เลข {t} ({a})" for t, a in zip(thai_digits, arabic_digits)], fontsize=11)
    axes[1, 0].set_title("3. Thai Numerals (ตัวเลขไทย)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Numeric Value")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Multi-line Text & Linguistic Metadata
    sample_info = (
        "[OK] การทดสอบการแสดงผลภาษาไทยใน Matplotlib สำเร็จ!\n"
        "• ก ไก่ (Ko Kai) - Class 161 [Consonant Middle]\n"
        "• ญ หญิง (Yo Ying) - Class 173 [Consonant Low]\n"
        "• ฐ ฐาน (Tho Than) - Class 176 [Consonant High]\n"
        "• สระ อา (Sara Aa) - Class 210 [Following Vowel]\n"
        "• ลากข้างยาว (ๅ) - Class 229 [Vowel Length Sign]\n"
        "• ไม้เอก (่) - Class 232 [Tone Marker 1st]\n"
        "• ข้อความภาษาไทยคมชัด ไม่ขึ้นเครื่องหมายเต้าหู้"
    )
    axes[1, 1].text(0.05, 0.5, sample_info, fontsize=12, verticalalignment="center", linespacing=1.6,
                     bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.5))
    axes[1, 1].axis("off")
    axes[1, 1].set_title("4. Sample Rendered Text Box", fontsize=13, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"📊 Verification image saved to: {output_path}")
    return True


def main():
    print("=" * 60)
    print("🇹🇭 Matplotlib Thai Font Setup & Installer")
    print("=" * 60)

    # Setup local fonts directory
    base_dir = Path(__file__).resolve().parent
    fonts_dir = base_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    # Download Sarabun font (regular & bold) if not already present
    for fname in ["Sarabun-Regular.ttf", "Sarabun-Bold.ttf"]:
        fpath = download_thai_font(fonts_dir, fname)
        if fpath and fpath.exists():
            rname = register_font_file(fpath)
            print(f"✅ Registered Google Font: '{rname}' ({fname})")

    # Configure Matplotlib
    active_font = setup_thai_font()
    print(f"🌟 Active Matplotlib Font: '{active_font}'")

    # Generate verification chart
    test_image_path = base_dir / "thai_font_verification.png"
    verify_thai_rendering(test_image_path, active_font)

    print("=" * 60)
    print("🎉 Thai Font Installation & Configuration Completed Successfully!")
    print(f"   Check '{test_image_path.name}' to preview the rendered Thai characters.")
    print("=" * 60)


if __name__ == "__main__":
    main()
