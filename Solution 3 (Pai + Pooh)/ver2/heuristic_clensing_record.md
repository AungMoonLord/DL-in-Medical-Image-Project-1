# Thai Character Dataset Cleansing Record (Round 2 vs Round 2 Cleaned)

This document records the manual and heuristic data cleansing applied to the **Thai Character Dataset** (`round2` $\to$ `round2-cleaned`).

### Cleansing Defect Taxonomy:
1. **Is Mislabeled**: Character sample belonged to a different Thai character class. Re-assigned and moved directly to the appropriate class folder.
2. **Is Fragmented**: Image contains severe visual degradation, truncated strokes, unintelligible noise, or bounding-box artifacts beyond reliable recognition. Excluded/purged from the dataset entirely.

### Mathematical Identity:
$$\text{Deletions} = \text{Is Mislabeled (Moved Out)} + \text{Is Fragmented (Purged)}$$
$$\text{New Samples Count} = \text{Original Sample Count} - \text{Deletions} + \text{Additions (Moved In)}$$

---

| TIS-620 Dec | Character | Category / English Name | Original Sample Count | Deletions | Additions | Is Mislabeled | Is Fragmented | New Samples Count |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 161 | ก | Ko Kai (Consonant) | 1,951 | 0 | 0 | 0 | 0 | 1,951 |
| 162 | ข | Kho Khai (Consonant) | 1,355 | 5 | 0 | 5 | 0 | 1,350 |
| 163 | ฃ | Kho Khuat (Obsolete Consonant - Rare) | 1 | 0 | 0 | 0 | 0 | 1 |
| 164 | ค | Kho Khwai (Consonant) | 1,448 | 1 | 8 | 1 | 0 | 1,455 |
| 167 | ง | Ngo Ngu (Consonant) | 1,078 | 0 | 0 | 0 | 0 | 1,078 |
| 168 | จ | Cho Chan (Consonant) | 1,202 | 0 | 0 | 0 | 0 | 1,202 |
| 169 | ฉ | Cho Ching (Consonant) | 43 | 0 | 0 | 0 | 0 | 43 |
| 170 | ช | Cho Chang (Consonant) | 1,013 | 3 | 12 | 3 | 0 | 1,022 |
| 171 | ซ | So So (Consonant) | 295 | 9 | 3 | 9 | 0 | 289 |
| 173 | ญ | Yo Ying (Consonant) | 113 | 0 | 1 | 0 | 0 | 114 |
| 175 | ฏ | To Patak (Consonant) | 32 | 0 | 0 | 0 | 0 | 32 |
| 176 | ฐ | Tho Than (Consonant) | 118 | 0 | 1 | 0 | 0 | 119 |
| 177 | ฑ | Tho Montho (Rare Consonant) | 1 | 0 | 0 | 0 | 0 | 1 |
| 178 | ฒ | Tho Phuthao (Consonant) | 48 | 0 | 0 | 0 | 0 | 48 |
| 179 | ณ | No Nen (Consonant) | 292 | 1 | 0 | 1 | 0 | 291 |
| 180 | ด | Do Dek (Consonant) | 2,025 | 7 | 2 | 7 | 0 | 2,020 |
| 181 | ต | To Tao (Consonant) | 1,662 | 9 | 6 | 7 | 2 | 1,659 |
| 182 | ถ | Tho Thung (Consonant) | 435 | 0 | 0 | 0 | 0 | 435 |
| 183 | ท | Tho Thahan (Consonant) | 1,745 | 0 | 0 | 0 | 0 | 1,745 |
| 184 | ธ | Tho Thong (Consonant) | 143 | 0 | 1 | 0 | 0 | 144 |
| 185 | น | No Nu (Consonant) | 4,863 | 3 | 4 | 3 | 0 | 4,864 |
| 186 | บ | Bo Baimai (Consonant) | 1,889 | 6 | 3 | 6 | 0 | 1,886 |
| 187 | ป | Po Pla (Consonant) | 1,133 | 0 | 6 | 0 | 0 | 1,139 |
| 188 | ผ | Pho Phueng (Consonant) | 344 | 0 | 0 | 0 | 0 | 344 |
| 189 | ฝ | Fo Fa (Consonant) | 39 | 0 | 0 | 0 | 0 | 39 |
| 190 | พ | Pho Phan (Consonant) | 729 | 0 | 1 | 0 | 0 | 730 |
| 191 | ฟ | Fo Fan (Consonant) | 103 | 0 | 0 | 0 | 0 | 103 |
| 192 | ภ | Pho Samphao (Consonant) | 239 | 0 | 0 | 0 | 0 | 239 |
| 193 | ม | Mo Ma (Consonant) | 3,306 | 6 | 2 | 6 | 0 | 3,302 |
| 194 | ย | Yo Yak (Consonant) | 2,197 | 9 | 0 | 7 | 2 | 2,188 |
| 195 | ร | Ro Ruea (Consonant) | 4,663 | 4 | 4 | 1 | 3 | 4,663 |
| 196 | ฤ | Rue (Vocalic Consonant) | 13 | 0 | 0 | 0 | 0 | 13 |
| 197 | ล | Lo Ling (Consonant) | 2,187 | 1 | 0 | 0 | 1 | 2,186 |
| 199 | ว | Wo Waen (Consonant) | 1,724 | 7 | 0 | 5 | 2 | 1,717 |
| 200 | ศ | So Sala (Consonant) | 94 | 1 | 0 | 1 | 0 | 93 |
| 201 | ษ | So Rusi (Consonant) | 264 | 0 | 2 | 0 | 0 | 266 |
| 202 | ส | So Suea (Consonant) | 1,595 | 3 | 0 | 3 | 0 | 1,592 |
| 203 | ห | Ho Hip (Consonant) | 1,490 | 1 | 0 | 1 | 0 | 1,489 |
| 204 | ฬ | Lo Chula (Consonant) | 3 | 0 | 0 | 0 | 0 | 3 |
| 205 | อ | O Ang (Consonant) | 3,272 | 0 | 1 | 0 | 0 | 3,273 |
| 206 | ฮ | Ho Nokhuk (Consonant) | 10 | 0 | 0 | 0 | 0 | 10 |
| 207 | ฯ | Paiyannoi (Ellipsis Symbol) | 20 | 0 | 0 | 0 | 0 | 20 |
| 209 |  | Mai Han-Akat (Vowel) | 4,120 | 0 | 5 | 0 | 0 | 4,125 |
| 210 | า | Sara Aa (Vowel) | 5,025 | 4,283* | 0* | 0* | 4,283* | 742 |
| 212 |  | Sara I (Vowel) | 47 | 0 | 0 | 0 | 0 | 47 |
| 213 |  | Sara Ii (Vowel) | 143 | 0 | 2 | 0 | 0 | 145 |
| 214 |  | Sara Ue (Vowel) | 14 | 0 | 0 | 0 | 0 | 14 |
| 215 |  | Sara Uee (Vowel) | 60 | 2 | 0 | 2 | 0 | 58 |
| 216 |  | Sara U (Vowel) | 155 | 0 | 0 | 0 | 0 | 155 |
| 217 |  | Sara Uu (Vowel) | 139 | 0 | 0 | 0 | 0 | 139 |
| 224 | เ | Sara E (Leading Vowel) | 203 | 1 | 0 | 0 | 1 | 202 |
| 225 | แ | Sara Ae (Leading Vowel) | 21 | 0 | 0 | 0 | 0 | 21 |
| 226 | โ | Sara O (Leading Vowel) | 699 | 0 | 0 | 0 | 0 | 699 |
| 227 | ใ | Sara Ai Maimuan (Leading Vowel) | 1,086 | 1 | 0 | 0 | 1 | 1,085 |
| 228 | ไ | Sara Ai Maimalai (Leading Vowel) | 725 | 0 | 0 | 0 | 0 | 725 |
| 229 | ๅ | Lakkhangyao (Vowel length marker) | 2,310 | 46** | 0** | 2** | 44** | 2,264 |
| 230 | ๆ | Maiyamok (Repetition Mark) | 121 | 0 | 0 | 0 | 0 | 121 |
| 231 |  | Maitaikhu (Tone/Vowel Mark) | 119 | 0 | 0 | 0 | 0 | 119 |
| 232 |  | Mai Ek (Tone Marker) | 479 | 0 | 2 | 0 | 0 | 481 |
| 233 |  | Mai Tho (Tone Marker) | 1,007 | 0 | 2 | 0 | 0 | 1,009 |
| 234 |  | Mai Tri (Tone Marker) | 49 | 0 | 0 | 0 | 0 | 49 |
| 236 |  | Thanthakhat / Karan (Silence Marker) | 726 | 0 | 0 | 0 | 0 | 726 |
| 240 | ๐ | Sun (Thai Digit 0) | 83 | 0 | 1 | 0 | 0 | 84 |
| 241 | ๑ | Nueng (Thai Digit 1) | 46 | 0 | 0 | 0 | 0 | 46 |
| 242 | ๒ | Song (Thai Digit 2) | 39 | 0 | 0 | 0 | 0 | 39 |
| 243 | ๓ | Sam (Thai Digit 3) | 23 | 0 | 0 | 0 | 0 | 23 |
| 244 | ๔ | Si (Thai Digit 4) | 16 | 0 | 0 | 0 | 0 | 16 |
| 245 | ๕ | Ha (Thai Digit 5) | 17 | 0 | 0 | 0 | 0 | 17 |
| 246 | ๖ | Hok (Thai Digit 6) | 12 | 0 | 0 | 0 | 0 | 12 |
| 247 | ๗ | Chet (Thai Digit 7) | 4 | 0 | 0 | 0 | 0 | 4 |
| 248 | ๘ | Paet (Thai Digit 8) | 27 | 0 | 0 | 0 | 0 | 27 |
| 249 | ๙ | Kao (Thai Digit 9) | 15 | 1 | 0 | 1 | 0 | 14 |
| **Total** | **72 Classes** | **Entire Dataset Summary** | **62,707** | **4,410** | **69** | **71** | **4,339** | **58,366** |

---

### 📌 Special Class Notes:
- **\*Class 210 (Sara Aa - า)**: Class 210 in `round2` suffered from extreme noise and unrepresentative bounding box fragments. A verified, high-quality subset of 742 pure samples was curated for training and evaluation. The remaining 4,283 noisy samples were excluded from the dataset.
- **\*\*Class 229 (Lakkhangyao - ๅ)**: Class 229 is visually identical in shape to Class 210 (า) except for tail elongation ratio. 2 samples were identified as true 210 and moved to Class 210, while 44 borderline/ambiguous samples with aspect ratios too similar to standard 210 were excluded to prevent decision boundary corruption.

---

### 🔄 Cross-Class Re-Assignment Ledger (All 71 Mislabeled Samples)

| Filename | Original Class | Source Char | Target Class | Target Char | Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `bc_009sg_1_25.jpg` | 162 | ข | 170 | ช | Reclassified to correct visual glyph class |
| `bc_019sg_6_450.jpg` | 162 | ข | 170 | ช | Reclassified to correct visual glyph class |
| `bl_008tg_1_33.jpg` | 162 | ข | 170 | ช | Reclassified to correct visual glyph class |
| `be_008sg_5_18.jpg` | 162 | ข | 209 |  | Reclassified to correct visual glyph class |
| `be_014sg_4_304.jpg` | 162 | ข | 209 |  | Reclassified to correct visual glyph class |
| `bc_014sg_2_26.jpg` | 164 | ค | 180 | ด | Reclassified to correct visual glyph class |
| `bl_007sg_8_343.jpg` | 170 | ช | 171 | ซ | Reclassified to correct visual glyph class |
| `bl_008sg_3_337.jpg` | 170 | ช | 171 | ซ | Reclassified to correct visual glyph class |
| `bl_011sg_3_7.jpg` | 170 | ช | 171 | ซ | Reclassified to correct visual glyph class |
| `bc_004sg_6_91.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `bc_008sg_14_160.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `bc_014sg_3_257.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `bc_015sg_3_282.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `bc_016sg_3_381.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `bc_017tg_3_55.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `bc_018sg_4_134.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `be_007sg_3_0.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `bl_011sg_4_362.jpg` | 171 | ซ | 170 | ช | Reclassified to correct visual glyph class |
| `be_020sg_2_104.jpg` | 179 | ณ | 173 | ญ | Reclassified to correct visual glyph class |
| `bl_009sg_3_400.jpg` | 180 | ด | 164 | ค | Reclassified to correct visual glyph class |
| `bc_019tg_7_248.jpg` | 180 | ด | 181 | ต | Reclassified to correct visual glyph class |
| `be_007tg_3_127.jpg` | 180 | ด | 181 | ต | Reclassified to correct visual glyph class |
| `be_008tg_5_130.jpg` | 180 | ด | 181 | ต | Reclassified to correct visual glyph class |
| `be_009sg_3_98.jpg` | 180 | ด | 181 | ต | Reclassified to correct visual glyph class |
| `be_014tg_2_126.jpg` | 180 | ด | 181 | ต | Reclassified to correct visual glyph class |
| `be_016tg_8_264.jpg` | 180 | ด | 181 | ต | Reclassified to correct visual glyph class |
| `bc_018sg_3_56.jpg` | 181 | ต | 164 | ค | Reclassified to correct visual glyph class |
| `bc_018sg_3_93.jpg` | 181 | ต | 164 | ค | Reclassified to correct visual glyph class |
| `bl_009sg_3_118.jpg` | 181 | ต | 164 | ค | Reclassified to correct visual glyph class |
| `bl_011sg_4_338.jpg` | 181 | ต | 164 | ค | Reclassified to correct visual glyph class |
| `bl_011tg_4_198.jpg` | 181 | ต | 164 | ค | Reclassified to correct visual glyph class |
| `bl_011tg_4_298.jpg` | 181 | ต | 164 | ค | Reclassified to correct visual glyph class |
| `bl_006sg_10_5.jpg` | 181 | ต | 180 | ด | Reclassified to correct visual glyph class |
| `bc_011sg_1_16.jpg` | 185 | น | 193 | ม | Reclassified to correct visual glyph class |
| `bc_008tg_14_291.jpg` | 185 | น | 209 |  | Reclassified to correct visual glyph class |
| `be_020tg_5_41.jpg` | 185 | น | 209 |  | Reclassified to correct visual glyph class |
| `be_003sg_7_105.jpg` | 186 | บ | 185 | น | Reclassified to correct visual glyph class |
| `be_003sg_9_303.jpg` | 186 | บ | 185 | น | Reclassified to correct visual glyph class |
| `be_013sg_5_37.jpg` | 186 | บ | 185 | น | Reclassified to correct visual glyph class |
| `bc_001sg_9_293.jpg` | 186 | บ | 187 | ป | Reclassified to correct visual glyph class |
| `bc_002sg_4_153.jpg` | 186 | บ | 187 | ป | Reclassified to correct visual glyph class |
| `be_018sg_6_37.jpg` | 186 | บ | 193 | ม | Reclassified to correct visual glyph class |
| `be_017tg_5_45.jpg` | 193 | ม | 185 | น | Reclassified to correct visual glyph class |
| `bl_003sg_6_36.jpg` | 193 | ม | 186 | บ | Reclassified to correct visual glyph class |
| `bl_010sg_6_335.jpg` | 193 | ม | 186 | บ | Reclassified to correct visual glyph class |
| `bl_010sg_6_411.jpg` | 193 | ม | 186 | บ | Reclassified to correct visual glyph class |
| `be_002sg_9_351.jpg` | 193 | ม | 209 |  | Reclassified to correct visual glyph class |
| `bl_005tg_3_58.jpg` | 193 | ม | 233 |  | Reclassified to correct visual glyph class |
| `bc_015sg_5_215.jpg` | 194 | ย | 187 | ป | Reclassified to correct visual glyph class |
| `bc_015sg_5_293.jpg` | 194 | ย | 187 | ป | Reclassified to correct visual glyph class |
| `bc_019sg_6_489.jpg` | 194 | ย | 187 | ป | Reclassified to correct visual glyph class |
| `bc_019sg_8_28.jpg` | 194 | ย | 187 | ป | Reclassified to correct visual glyph class |
| `be_018tg_11_26.jpg` | 194 | ย | 201 | ษ | Reclassified to correct visual glyph class |
| `be_019sg_4_32.jpg` | 194 | ย | 201 | ษ | Reclassified to correct visual glyph class |
| `bl_003tg_1_11.jpg` | 194 | ย | 205 | อ | Reclassified to correct visual glyph class |
| `bl_004sg_10_14.jpg` | 195 | ร | 184 | ธ | Reclassified to correct visual glyph class |
| `bl_001sg_3_58.jpg` | 199 | ว | 195 | ร | Reclassified to correct visual glyph class |
| `bl_002sg_3_93.jpg` | 199 | ว | 195 | ร | Reclassified to correct visual glyph class |
| `bl_005sg_3_125.jpg` | 199 | ว | 195 | ร | Reclassified to correct visual glyph class |
| `bl_007sg_3_219.jpg` | 199 | ว | 195 | ร | Reclassified to correct visual glyph class |
| `bl_005sg_11_9.jpg` | 199 | ว | 240 | ๐ | Reclassified to correct visual glyph class |
| `bc_019sg_6_131.jpg` | 200 | ศ | 164 | ค | Reclassified to correct visual glyph class |
| `bc_016tg_6_8.jpg` | 202 | ส | 176 | ฐ | Reclassified to correct visual glyph class |
| `be_008sg_4_341.jpg` | 202 | ส | 232 |  | Reclassified to correct visual glyph class |
| `be_013tg_5_40.jpg` | 202 | ส | 232 |  | Reclassified to correct visual glyph class |
| `bc_004sg_4_207.jpg` | 203 | ห | 190 | พ | Reclassified to correct visual glyph class |
| `bc_001tg_8_72.jpg` | 215 |  | 213 |  | Reclassified to correct visual glyph class |
| `bc_007tg_5_160.jpg` | 215 |  | 213 |  | Reclassified to correct visual glyph class |
| `ce_001sg_2_12.jpg` | 229 | ๅ | 210 | า | Reclassified to correct visual glyph class |
| `ce_001sg_2_8.jpg` | 229 | ๅ | 210 | า | Reclassified to correct visual glyph class |
| `be_004sg_7_303.jpg` | 249 | ๙ | 233 |  | Reclassified to correct visual glyph class |
