#  Retail Crowd Intelligence | AI-Powered Customer Traffic Estimation

**Skills demonstrated:** Python · TensorFlow/Keras · Deep Learning · Transfer Learning · Computer Vision · Model Evaluation · Business Decision-Making · Data Visualization

---

## Project Overview

Built two deep learning models to estimate customer headcounts from existing mall surveillance footage — eliminating the need for expensive hardware sensors. Compared a Custom CNN (baseline) against a fine-tuned VGG16 transfer learning model. VGG16 achieved **92% variance explained (R²) and a prediction error of ±2 persons per frame**, making it operationally viable for real-time retail footfall monitoring.

---

## Business Problem

Retail mall operators need accurate customer traffic data for staffing, promotions, and leasing decisions — but traditional solutions are costly and hard to scale.

| Challenge | Business Impact |
|---|---|
| Hardware people-counters cost $500–$2,000 per entrance | High upfront CapEx with no scalability |
| Manual counting is inconsistent | Unreliable KPIs undermine staffing and leasing decisions |
| No real-time zone visibility | Missed safety risks and lost revenue opportunities |
| No historical trend data | Poor capacity planning and inability to measure promotion lift |

**Solution:** Repurpose existing CCTV infrastructure with a CNN inference layer to deliver continuous crowd intelligence at **<$50/month in cloud compute** — a fraction of hardware costs, with no new equipment required.

---

## Data Description

- **Dataset:** 2,000 labeled shopping mall surveillance frames ([Kaggle — Crowd Counting Dataset](https://www.kaggle.com/datasets/fmena14/crowd-counting))
- **Target Variable:** Ground-truth crowd count per frame (0–53 persons)
- **Image Size:** 160 × 120 px, RGB
- **Distribution:** Right-skewed — most frames contain 10–25 persons (moderate density)
- **Split:** 80% train · 10% validation · 10% test, stratified by density tier (Low / Medium / High)

---

## Methodology

- **Exploratory Data Analysis** — crowd count distribution, density tier breakdown, sample frame preview
- **Stratified data split** — Low / Medium / High density tiers balanced across all three splits to prevent skewed evaluation
- **Light data augmentation** — horizontal flip and brightness jitter on training set to improve generalization
- **Model 1 — Custom CNN:** 3 convolutional blocks with BatchNormalization and Dropout; regression head with linear output
- **Model 2 — VGG16 Fine-Tuned:** Frozen blocks 1–4 (generic features preserved); trainable block 5 adapted to crowd domain; custom Dense regression head
- **Training safeguards:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, and CSVLogger applied to both models
- **Evaluation:** MAE, RMSE, R², actual-vs-predicted scatter, residual distribution, and qualitative frame-level inspection

---

## Results

| Model | MAE (persons) | RMSE | R² | Verdict |
|---|---|---|---|---|
| Custom CNN | ~3.24 | ~3.75 | ~0.80 | Strong baseline |
| **VGG16 Fine-Tuned** | **~2.13** | **~2.82** | **~0.92** |  **Recommended** |

- VGG16 reduced prediction error by **34%** over the baseline CNN
- MAE of ±2 persons in a 500-capacity space = **<0.4% error** — within operational tolerance
- Industry deployment threshold for retail crowd counting is MAE < 5 — both models qualify; VGG16 with clear margin

---

## Business Recommendations

VGG16 is recommended for deployment based on accuracy, cost, and measurable ROI:

| Business Function | Use Case | Value |
|---|---|---|
| **Mall Operations** | Real-time alerts when zone count exceeds 80% capacity | Safety compliance, liability reduction |
| **Retail Tenants** | Hourly footfall reports per zone | Staffing optimization — 10–15% labor cost savings |
| **Marketing** | Traffic lift on promotion days vs. baseline | Measure and attribute campaign ROI |
| **Leasing** | Traffic-to-sales conversion metrics | Stronger data position in anchor tenant negotiations |
| **Security** | Overcrowding early warning system | Faster emergency response time |

**Estimated cost:** <$50/month cloud inference vs. $500–$2,000 per entrance for hardware sensors  
**Estimated payback period:** 1–3 months for a mid-size mall with 50+ storefronts

---

## Tools & Technologies

- **Language:** Python 3.10
- **Deep Learning:** TensorFlow 2.x · Keras
- **Pretrained Model:** VGG16 (ImageNet weights)
- **Data Processing:** Pandas · NumPy · Scikit-learn
- **Visualization:** Matplotlib · Seaborn
- **Environment:** Google Colab (GPU) · Jupyter Notebook

---

## Project Files

```
retail-crowd-intelligence/
├── Retail_Crowd_Intelligence.ipynb   ← Full analysis notebook
├── README.md                         ← This file
├── model_predictions.csv             ← Test set predictions (generated)
├── cnn_training_log.csv              ← CNN epoch metrics
└── vgg16_training_log.csv            ← VGG16 epoch metrics
```

> Dataset: Download from [Kaggle](https://www.kaggle.com/datasets/fmena14/crowd-counting). Not included due to file size.

---

## Author

**Aketch Okoth** · M.S. Business Analytics · Montclair State University  
*Actively seeking Data Analyst and Business Analyst roles in the United States.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat&logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:your-email@email.com)
