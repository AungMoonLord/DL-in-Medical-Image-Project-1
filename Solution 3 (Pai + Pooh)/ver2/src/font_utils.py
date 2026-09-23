"""
Font utilities for Thai character visualization in Matplotlib (ver2).

Provides automatic detection, download (Google Fonts Sarabun / Noto Sans Thai),
Matplotlib font registration, and rcParams configuration to eliminate tofu (□□□) boxes.
"""

import os
from pathlib import Path
import sys
from typing import List, Optional
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Recommended Thai font candidates in order of preference
SYSTEM_THAI_FONT_CANDIDATES = [
    "Leelawadee UI",
    "Leelawadee",
    "Tahoma",
    "Cordia New",
    "Angsana New",
    "Sarabun",
    "Noto Sans Thai",
    "KodchiangUPC",
    "JasmineUPC",
    "CordiaUPC",
    "AngsanaUPC",
]

ONLINE_FONT_URLS = {
    "Sarabun-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sarabun/Sarabun-Regular.ttf",
    "Sarabun-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sarabun/Sarabun-Bold.ttf",
    "NotoSansThai-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansthai/NotoSansThai-Regular.ttf",
}


def find_system_thai_font() -> Optional[str]:
    """Finds an available Thai font already registered in Matplotlib."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in SYSTEM_THAI_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return None


def download_thai_font(target_dir: Optional[Path] = None, font_name: str = "Sarabun-Regular.ttf") -> Optional[Path]:
    """
    Downloads a clean open-source Thai font from Google Fonts to a local directory.
    """
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent / "fonts"
    target_dir.mkdir(parents=True, exist_ok=True)
    font_path = target_dir / font_name

    if font_path.exists() and font_path.stat().st_size > 1000:
        return font_path

    url = ONLINE_FONT_URLS.get(font_name)
    if not url:
        return None

    try:
        print(f"📥 Downloading Thai font '{font_name}' from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response, open(font_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"✅ Successfully downloaded to: {font_path}")
        return font_path
    except Exception as e:
        print(f"⚠️ Could not download font from internet ({e}). Falling back to system fonts.")
        return None


def register_font_file(font_path: Path) -> str:
    """Registers a TTF/OTF font file directly into Matplotlib's font manager."""
    fm.fontManager.addfont(str(font_path))
    prop = fm.FontProperties(fname=str(font_path))
    font_name = prop.get_name()
    return font_name


def setup_thai_font(preferred_font: Optional[str] = None) -> str:
    """
    Configures Matplotlib to render Thai characters correctly.
    
    1. Checks if a local font file exists in fonts/ or downloads Sarabun.
    2. Fallback to system fonts (Leelawadee UI, Tahoma, etc.).
    3. Sets rcParams['font.family'] and rcParams['axes.unicode_minus'] = False.
    
    Returns:
        The active font family name.
    """
    active_font = None

    # 1. If explicit font provided and registered
    if preferred_font:
        active_font = preferred_font

    # 2. Check local fonts directory in project
    fonts_dir = Path(__file__).resolve().parent.parent / "fonts"
    if fonts_dir.exists():
        for ttf_file in fonts_dir.glob("*.ttf"):
            try:
                active_font = register_font_file(ttf_file)
                break
            except Exception:
                pass

    # 3. If no local font loaded, try finding system Thai font
    if not active_font:
        active_font = find_system_thai_font()

    # 4. If still none found, download Sarabun
    if not active_font:
        dl_path = download_thai_font(fonts_dir, "Sarabun-Regular.ttf")
        if dl_path and dl_path.exists():
            try:
                active_font = register_font_file(dl_path)
            except Exception:
                pass

    # Fallback to sans-serif
    if not active_font:
        active_font = "sans-serif"

    # Set Matplotlib configurations
    plt.rcParams["font.family"] = active_font
    plt.rcParams["axes.unicode_minus"] = False
    
    # Also add Thai fallback fonts into font.sans-serif list
    sans_list = plt.rcParams.get("font.sans-serif", [])
    if isinstance(sans_list, str):
        sans_list = [sans_list]
    for candidate in SYSTEM_THAI_FONT_CANDIDATES:
        if candidate not in sans_list:
            sans_list.insert(0, candidate)
    if active_font not in sans_list:
        sans_list.insert(0, active_font)
    plt.rcParams["font.sans-serif"] = sans_list

    return active_font
