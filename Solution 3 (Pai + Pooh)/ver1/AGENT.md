# Thai Character Classification (Solution 3: Pai + Pooh) - System Context & Blueprint (`ver1`)

## 🎯 Project Objective & Overview
This project develops a high-performance deep learning pipeline for **72-class Thai Character Classification** using:
- **Custom GlyphCNNs & Adapted Transfer Learning backbones** (ResNet-18, MobileNetV3-Small)
- **Domain-specific Thai glyph augmentations** (Morphological stroke dilation/erosion, bounded rotation)
- **Zero-leakage deterministic 80/20 train-test splitting** (Grouped by document source and page)
- **Adaptive learning rate schedules & optimization** (Warmup + Cosine Annealing, Differential AdamW)
- **Class-imbalance loss engineering** (Class-Balanced Focal Loss, Label Smoothing, Balanced Sampling)

---

## 📂 System Paths & Environment
- **Active Workspace**: `D:\vscode\kmitl\3-1\dlmed\DL-in-Medical-Image-Project-1\Solution 3 (Pai + Pooh)\ver1`
- **Dataset Location**: `D:\vscode\kmitl\3-1\dlmed\ThaiCharacter Dataset\round2`
- **Hardware Profile**: NVIDIA GeForce GTX 1650 (4 GB VRAM), CUDA 11.8 / 13.1
- **Course Context**: KMITL Deep Learning in Medical Images (Project 1)

---

## 📊 Dataset Metadata & TIS-620 Class Mapping

The dataset contains **62,707 images** across **72 classes**. Folder names correspond to the decimal byte value of the TIS-620 encoding.

| TIS-620 Dec | Hex | Character | Category / English Name | Total Samples | Train (80%) | Test (20%) |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 161 | 0xA1 | ก | Ko Kai (Consonant) | 1,952 | 1,556 | 396 |
| 162 | 0xA2 | ข | Kho Khai (Consonant) | 1,355 | 1,079 | 276 |
| 163 | 0xA3 | ฃ | Kho Khuat (Obsolete Consonant - Rare) | 1 | 1 | 0* |
| 164 | 0xA4 | ค | Kho Khwai (Consonant) | 1,448 | 1,153 | 295 |
| 167 | 0xA7 | ง | Ngo Ngu (Consonant) | 1,078 | 859 | 219 |
| 168 | 0xA8 | จ | Cho Chan (Consonant) | 1,202 | 957 | 245 |
| 169 | 0xA9 | ฉ | Cho Ching (Consonant) | 43 | 34 | 9 |
| 170 | 0xAA | ช | Cho Chang (Consonant) | 1,013 | 806 | 207 |
| 171 | 0xAB | ซ | So So (Consonant) | 295 | 235 | 60 |
| 173 | 0xAD | ญ | Yo Ying (Consonant) | 113 | 90 | 23 |
| 175 | 0xAF | ฏ | To Patak (Consonant) | 32 | 25 | 7 |
| 176 | 0xB0 | ฐ | Tho Than (Consonant) | 118 | 94 | 24 |
| 177 | 0xB1 | ฑ | Tho Montho (Rare Consonant) | 1 | 1 | 0* |
| 178 | 0xB2 | ฒ | Tho Phuthao (Consonant) | 48 | 38 | 10 |
| 179 | 0xB3 | ณ | No Nen (Consonant) | 292 | 232 | 60 |
| 180 | 0xB4 | ด | Do Dek (Consonant) | 2,025 | 1,612 | 413 |
| 181 | 0xB5 | ต | To Tao (Consonant) | 1,662 | 1,323 | 339 |
| 182 | 0xB6 | ถ | Tho Thung (Consonant) | 435 | 346 | 89 |
| 183 | 0xB7 | ท | Tho Thahan (Consonant) | 1,745 | 1,389 | 356 |
| 184 | 0xB8 | ธ | Tho Thong (Consonant) | 143 | 114 | 29 |
| 185 | 0xB9 | น | No Nu (Consonant) | 4,863 | 3,871 | 992 |
| 186 | 0xBA | บ | Bo Baimai (Consonant) | 1,889 | 1,504 | 385 |
| 187 | 0xBB | ป | Po Pla (Consonant) | 1,133 | 902 | 231 |
| 188 | 0xBC | ผ | Pho Phueng (Consonant) | 344 | 274 | 70 |
| 189 | 0xBD | ฝ | Fo Fa (Consonant) | 39 | 31 | 8 |
| 190 | 0xBE | พ | Pho Phan (Consonant) | 729 | 580 | 149 |
| 191 | 0xBF | ฟ | Fo Fan (Consonant) | 103 | 82 | 21 |
| 192 | 0xC0 | ภ | Pho Samphao (Consonant) | 239 | 190 | 49 |
| 193 | 0xC1 | ม | Mo Ma (Consonant) | 3,306 | 2,631 | 675 |
| 194 | 0xC2 | ย | Yo Yak (Consonant) | 2,197 | 1,749 | 448 |
| 195 | 0xC3 | ร | Ro Ruea (Consonant) | 4,663 | 3,712 | 951 |
| 196 | 0xC4 | ฤ | Rue (Vocalic Consonant) | 13 | 10 | 3 |
| 197 | 0xC5 | ล | Lo Ling (Consonant) | 2,187 | 1,741 | 446 |
| 199 | 0xC7 | ว | Wo Waen (Consonant) | 1,724 | 1,372 | 352 |
| 200 | 0xC8 | ศ | So Sala (Consonant) | 94 | 75 | 19 |
| 201 | 0xC9 | ษ | So Rusi (Consonant) | 264 | 210 | 54 |
| 202 | 0xCA | ส | So Suea (Consonant) | 1,595 | 1,270 | 325 |
| 203 | 0xCB | ห | Ho Hip (Consonant) | 1,490 | 1,186 | 304 |
| 204 | 0xCC | ฬ | Lo Chula (Consonant) | 3 | 2 | 1 |
| 205 | 0xCD | อ | O Ang (Consonant) | 3,272 | 2,604 | 668 |
| 206 | 0xCE | ฮ | Ho Nokhuk (Consonant) | 10 | 8 | 2 |
| 207 | 0xCF | ฯ | Paiyannoi (Ellipsis Symbol) | 20 | 16 | 4 |
| 209 | 0xD1 | ั | Mai Han-Akat (Vowel) | 4,120 | 3,280 | 840 |
| 210 | 0xD2 | า | Sara Aa (Vowel) | 5,025 | 4,000 | 1,025 |
| 212 | 0xD4 | ิ | Sara I (Vowel) | 47 | 37 | 10 |
| 213 | 0xD5 | ี | Sara Ii (Vowel) | 143 | 114 | 29 |
| 214 | 0xD6 | ึ | Sara Ue (Vowel) | 14 | 11 | 3 |
| 215 | 0xD7 | ื | Sara Uee (Vowel) | 60 | 48 | 12 |

---

| 216 | 0xD8 | ุ | Sara U (Vowel) | 155 | 123 | 32 |
| 217 | 0xD9 | ู | Sara Uu (Vowel) | 139 | 111 | 28 |
| 224 | 0xE0 | เ | Sara E (Leading Vowel) | 203 | 162 | 41 |
| 225 | 0xE1 | แ | Sara Ae (Leading Vowel) | 21 | 17 | 4 |
| 226 | 0xE2 | โ | Sara O (Leading Vowel) | 699 | 556 | 143 |
| 227 | 0xE3 | ใ | Sara Ai Maimuan (Leading Vowel) | 1,086 | 864 | 222 |
| 228 | 0xE4 | ไ | Sara Ai Maimalai (Leading Vowel) | 725 | 577 | 148 |
| 229 | 0xE5 | ๅ | Lakkhangyao (Vowel length marker) | 2,310 | 1,839 | 471 |
| 230 | 0xE6 | ๆ | Maiyamok (Repetition Mark) | 121 | 96 | 25 |
| 231 | 0xE7 | ็ | Maitaikhu (Tone/Vowel Mark) | 119 | 95 | 24 |
| 232 | 0xE8 | ่ | Mai Ek (Tone Marker) | 479 | 381 | 98 |
| 233 | 0xE9 | ้ | Mai Tho (Tone Marker) | 1,007 | 802 | 205 |

---

| 234 | 0xEA | ๊ | Mai Tri (Tone Marker) | 49 | 39 | 10 |
| 236 | 0xEC | ์ | Thanthakhat / Karan (Silence Marker) | 726 | 578 | 148 |
| 240 | 0xF0 | ๐ | Sun (Thai Digit 0) | 83 | 66 | 17 |
| 241 | 0xF1 | ๑ | Nueng (Thai Digit 1) | 46 | 37 | 9 |
| 242 | 0xF2 | ๒ | Song (Thai Digit 2) | 39 | 31 | 8 |
| 243 | 0xF3 | ๓ | Sam (Thai Digit 3) | 23 | 18 | 5 |
| 244 | 0xF4 | ๔ | Si (Thai Digit 4) | 16 | 13 | 3 |
| 245 | 0xF5 | ๕ | Ha (Thai Digit 5) | 17 | 14 | 3 |
| 246 | 0xF6 | ๖ | Hok (Thai Digit 6) | 12 | 10 | 2 |
| 247 | 0xF7 | ๗ | Chet (Thai Digit 7) | 4 | 3 | 1 |
| 248 | 0xF8 | ๘ | Paet (Thai Digit 8) | 27 | 21 | 6 |
| 249 | 0xF9 | ๙ | Kao (Thai Digit 9) | 16 | 13 | 3 |

*\*Note: Classes with $N=1$ (163, 177) are placed in Train, with synthetic font rendering available for out-of-sample test evaluation.*

---

## 🛡️ Data Leakage Prevention Rules
1. **Group-Page Partitioning**: Characters from the same document source (`doc_id`) and scanned page (`page_id`) must remain in the same split.
2. **Counterpart Lock**: Source (`_sg`) and target (`_tg`) scan pairs are bound together under the same hash partition.
3. **Deterministic Hash**: Partitioning is calculated using `SHA-256(class_id + '_' + group_key) % 100 < 80` to prevent seed drift.

---

## 🏗️ Model Architecture Decisions
- **CustomGlyphCNN**: 4-Stage Conv-BN-Mish-SE with native $32 \times 32$ / $64 \times 64$ receptive field.
- **Adapted ResNet-18**: Stem modified from $7 \times 7$ conv (stride 2) + maxpool to $3 \times 3$ conv (stride 1) with no initial pooling.
- **Adapted MobileNetV3-Small**: High parameter efficiency with inverted residual bottlenecks and SE attention.

---

## 🚀 Training & Optimization Hyperparameters
- **Input Resolution**: $32 \times 32$ (Custom CNN) or $64 \times 64$ (Transfer Learning) with aspect-preserving letterbox padding.
- **Batch Size**: 64 (well within the 4GB GTX 1650 memory limit).
- **Optimizer**: AdamW (Weight Decay = 0.01).
- **LR Policy**: 3-epoch Linear Warmup + Cosine Annealing decay.
- **Loss**: Class-Balanced Focal Loss ($\beta=0.999$, $\gamma=2.0$) + Label Smoothing ($\epsilon=0.08$).
- **Precision**: Mixed Precision (`torch.amp.autocast('cuda')`).
