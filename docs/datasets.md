# Supported Datasets

ITKIT provides dataset restructuring scripts to convert datasets from their official release formats to a consistent ITKIT structure. This enables unified API usage across different datasets.

## Conversion Scripts

For each supported dataset, you can find conversion scripts at: `itkit/dataset/<dataset_name>/convert_<format>.py`

## Dataset List

### 1. AbdomenCT-1K

**Description:** Large-scale abdominal organ segmentation dataset.

**Organs:** Multiple abdominal organs

**Modality:** CT

**Reference:** J. Ma et al., "AbdomenCT-1K: Is Abdominal Organ Segmentation a Solved Problem?," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6695-6714, 1 Oct. 2022.

**DOI:** 10.1109/TPAMI.2021.3100536

**Link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/9497733)

---

### 2. BraTS 2024

**Description:** Brain Tumor Segmentation Challenge - Glioma segmentation on post-treatment MRI.

**Organs:** Brain, tumor subregions

**Modality:** MRI (multi-sequence)

**Reference:** Maria Correia de Verdier, et al., "The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI," arXiv preprint arXiv:2405.18368, 2024.

**arXiv:** [2405.18368](https://arxiv.org/abs/2405.18368)

---

### 3. CT-ORG

**Description:** Multiple organ segmentation in computed tomography.

**Organs:** Liver, bladder, lungs, kidneys, bone, brain

**Modality:** CT

**Reference:** Rister, B., Yi, D., Shivakumar, K. et al. CT-ORG, a new dataset for multiple organ segmentation in computed tomography. Sci Data 7, 381 (2020).

**DOI:** [10.1038/s41597-020-00715-8](https://doi.org/10.1038/s41597-020-00715-8)

**Link:** [Nature](https://www.nature.com/articles/s41597-020-00715-8)

---

### 4. CTSpine1K

**Description:** Large-scale dataset for spinal vertebrae segmentation.

**Organs:** Vertebrae

**Modality:** CT

**Reference:** Yang Deng, Ce Wang, Yuan Hui, et al. CtSpine1k: A large-scale dataset for spinal vertebrae segmentation in computed tomography. arXiv preprint arXiv:2105.14711 (2021).

**arXiv:** [2105.14711](https://arxiv.org/pdf/2105.14711)

---

### 5. FLARE 2022

**Description:** Fast and Low-resource semi-supervised Abdominal oRgan sEgmentation.

**Organs:** Liver, kidneys, spleen, pancreas

**Modality:** CT

**Reference:** Jun Ma, et al., Unleashing the Strengths of Unlabeled Data in Pan-cancer Abdominal Organ Quantification: the FLARE22 Challenge. arXiv preprint arXiv:2308.05862, 2023.

**arXiv:** [2308.05862](https://arxiv.org/abs/2308.05862)

---

### 6. FLARE 2023

**Description:** Fast, Low-resource, and Accurate Organ and Pan-cancer Segmentation in Abdomen CT.

**Organs:** Multiple abdominal organs and tumors

**Modality:** CT

**Reference:** Jun Ma, Bo Wang (Eds.). Fast, Low-resource, and Accurate Organ and Pan-cancer Segmentation in Abdomen CT: MICCAI Challenge, FLARE 2023, Held in Conjunction with MICCAI 2023, Vancouver, BC, Canada, October 8, 2023, Proceedings. Lecture Notes in Computer Science. Springer, Cham, 2024.

**DOI:** [10.1007/978-3-031-58776-4](https://doi.org/10.1007/978-3-031-58776-4)

**Link:** [Springer](https://link.springer.com/book/10.1007/978-3-031-58776-4)

---

### 7. ImageTBAD

**Description:** 3D Computed Tomography Angiography Image Dataset for Automatic Segmentation of Type-B Aortic Dissection.

**Organs:** Aorta

**Modality:** CTA

**Reference:** Yao Z, Xie W, Zhang J, Dong Y, Qiu H, Yuan H, Jia Q, Wang T, Shi Y, Zhuang J, Que L, Xu X and Huang M (2021) ImageTBAD: A 3D Computed Tomography Angiography Image Dataset for Automatic Segmentation of Type-B Aortic Dissection. Front. Physiol. 12:732711.

**DOI:** [10.3389/fphys.2021.732711](https://doi.org/10.3389/fphys.2021.732711)

**Link:** [Frontiers](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.732711/full)

---

### 8. KiTS23

**Description:** Kidney and Kidney Tumor Segmentation Challenge.

**Organs:** Kidneys, renal tumors, renal cysts

**Modality:** CT (corticomedullary phase)

**References:**
1. Nicholas Heller, Fabian Isensee, Dasha Trofimova, et al. The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT. arXiv:2307.01984 [cs.CV], 2023.
2. Nicholas Heller, Fabian Isensee, Klaus H. Maier-Hein, et al. The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge. Medical Image Analysis, Vol. 67, Article 101821, 2021.

**Website:** [kits-challenge.org](https://kits-challenge.org/kits23/)

**DOI:** [10.1016/j.media.2020.101821](https://doi.org/10.1016/j.media.2020.101821)

---

### 9. LUNA16

**Description:** Lung Nodule Analysis - automatic detection of pulmonary nodules.

**Organs:** Lungs, nodules

**Modality:** CT

**Reference:** Arnaud Arindra Adiyoso Setio, Alberto Traverso, Thomas de Bel, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. Medical Image Analysis, Vol. 42, pp. 1–13, 2017.

**DOI:** [10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015)

**Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1361841517301020)

---

### 10. TCGA

**Description:** The Cancer Genome Atlas - comprehensive cancer genomics dataset.

**Organs:** Various cancer types

**Modality:** Multiple

**Website:** [cancer.gov/TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)

---

### 11. TotalSegmentator

**Description:** Robust segmentation of 104 anatomic structures in CT images.

**Organs:** 104 anatomical structures

**Modality:** CT

**Reference:** Wasserthal Jakob, et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence, 5, 5, 2023.

**DOI:** [10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024)

**Link:** [RSNA](https://pubs.rsna.org/doi/10.1148/ryai.230024)

---

### 12. LiTS

**Description:** Liver Tumor Segmentation Benchmark.

**Organs:** Liver, liver tumors

**Modality:** CT

**Reference:** Bilic, Patrick and Christ, Patrick and Li, Hongwei Bran and Vorontsov, Eugene and Ben-Cohen, Avi and Kaissis, Georgios and Szeskin, Adi and Jacobs, Colin and Mamani, Gabriel Efrain Humpire and Chartrand, Gabriel and others. The liver tumor segmentation benchmark (lits). Medical Image Analysis, volume 84, 2023, 102680.

**DOI:** [10.1016/j.media.2022.102680](https://doi.org/10.1016/j.media.2022.102680)

**Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1361841522003085)

---

## Dataset Categories

### By Modality

**CT Datasets:**
- AbdomenCT-1K
- CT-ORG
- CTSpine1K
- FLARE 2022/2023
- ImageTBAD
- KiTS23
- LUNA16
- TotalSegmentator
- LiTS

**MRI Datasets:**
- BraTS 2024

**Multi-Modal:**
- TCGA

### By Anatomical Region

**Abdomen:**
- AbdomenCT-1K
- FLARE 2022/2023
- LiTS

**Brain:**
- BraTS 2024
- CT-ORG (brain)

**Thorax:**
- LUNA16
- CT-ORG (lungs)

**Kidney:**
- KiTS23
- CT-ORG (kidneys)

**Cardiovascular:**
- ImageTBAD

**Spine:**
- CTSpine1K

**Whole Body:**
- TotalSegmentator

## Using Conversion Scripts

### Example: Converting AbdomenCT-1K

```bash
# Navigate to dataset conversion directory
cd itkit/dataset/AbdomenCT-1K/

# Run conversion script
python convert_official.py \
    --input /path/to/official/dataset \
    --output /path/to/itkit/format
```

### General Workflow

1. **Download official dataset** from the source
2. **Locate conversion script** in `itkit/dataset/<dataset_name>/`
3. **Run conversion script** with appropriate paths
4. **Verify structure** matches ITKIT format (image/ and label/ folders)
5. **Use with ITKIT tools** for preprocessing and training

## Custom Dataset Preparation

If your dataset is not in the list, you can manually convert it to ITKIT format:

1. Create `image/` and `label/` folders
2. Place image files in `image/`
3. Place corresponding label files in `label/` with matching names
4. Use any supported format (.mha, .nii.gz, etc.)
5. Optionally create `meta.json` with dataset information

## Contributing New Datasets

If you've created a conversion script for a new dataset, please consider contributing it to ITKIT:

1. Create conversion script following the existing pattern
2. Add dataset documentation
3. Submit a pull request

See [Contributing Guide](contributing.md) for details.

## Next Steps

- [Dataset Structure](dataset_structure.md) - Understand ITKIT dataset format
- [Preprocessing](preprocessing.md) - Learn preprocessing tools
- [Models](models.md) - Explore available models for training
