# Harnessing Social Media Data for Natural Disaster Information

An automated two-stage natural language processing (NLP) framework designed to identify and categorize disaster-related information from social media (specifically Reddit) during extreme weather events. This project serves as the codebase for the research paper submitted to *Natural Hazards*.

It features a binary classification gatekeeper (Stage 1) to filter out noise, a fine-grained multi-class classifier (Stage 2) to categorize actionable updates, and an interactive Streamlit web dashboard for real-time inference and monitoring.

🌐 **Live Deployed Web Application:** [Streamlit Live Demo](https://harnessing-social-media-data-for-natural-disaster-information.streamlit.app/)

---

## 🌪️ Project Overview

During natural disasters (e.g., hurricanes and wildfires), social media platforms provide crucial, real-time, localized information that can assist emergency management. However, these feeds contain high levels of noise (memes, opinions, general chatter). 

This project implements a **two-stage classification pipeline**:
1. **Stage 1 (Informational vs. Non-Informational):** Filters out non-disaster chatter.
2. **Stage 2 (Fine-Grained Classification):** Categorizes the informational posts into six critical categories:
   *   **Weather Updates:** Forecasts, storm tracks, wind speeds, meteorological data.
   *   **Geolocation Info:** Spatial details, specific locations impacted, and coordinates.
   *   **Evacuation & Safety:** Evacuation zones, shelter locations, safety advisories.
   *   **Damage Reports:** Infrastructure damage, power outages, road closures, flooding.
   *   **Community Support:** Mutual aid coordination, volunteers, resource sharing.
   *   **Official Announcements:** Institutional updates from government or official bodies (e.g., NOAA).

---

## 📂 Project Structure

```directory
├── Title-Category.csv                 # Dataset for Stage 1 (Binary classifier)
├── Hurricane_Reddit_Categorized.xlsx   # Dataset for Stage 2 (Fine-grained classifier)
├── train_models.py                    # Script to train Stage 1 and Stage 2 BERT models
├── hurricane_classifier_ui.py         # Streamlit-based web dashboard interface
├── requirements.txt                   # Project package dependencies
└── saved_models/                      # Folder containing trained models and encoders
    ├── stage1_info_model/             # Saved Stage 1 BERT model
    ├── stage2_category_model/         # Saved Stage 2 BERT model
    └── stage2_label_encoder.pkl       # Label encoder for the 6 category classes
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. A GPU is recommended but not required for running inference or evaluating models.

### 2. Clone the Repository
```bash
git clone https://github.com/bandisumanth1811/Harnessing-Social-Media-Data-for-Natural-Disaster-Information.git
cd Harnessing-Social-Media-Data-for-Natural-Disaster-Information
```

### 3. Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

You can access the **Live Cloud Application** directly at:
👉 **[Streamlit Live Demo URL](https://harnessing-social-media-data-for-natural-disaster-information.streamlit.app/)**

Alternatively, if you want to run the project locally (e.g., for local development, offline usage, or testing models you retrained on your own computer), follow the instructions below:

### 🖥️ 1. Start the Streamlit Web Application Locally
Run the interactive dashboard on your local machine to classify post titles in real-time:
```bash
streamlit run hurricane_classifier_ui.py
```
Once launched, the terminal will provide a URL, typically `http://localhost:8501`, which you can open in your web browser. This local version will load model weights stored locally in your `saved_models/` folder.

### 🏋️ 2. Train the Models
To retrain both Stage 1 and Stage 2 models on your datasets:
```bash
python train_models.py
```
*This trains the models using the HuggingFace BERT Transformer model (`bert-base-uncased`) and saves the weights to the `saved_models/` folder.*

---

## 📊 Dataset Characteristics

The models are trained and validated on curated Reddit titles collected during major hurricane events:

| Dataset / Sub-corpus | Samples | Categories | Avg. Title Length (words) | Vocabulary Size |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 1 Corpus (Raw Titles)** | 160 | 2 | 10.29 | 410 |
| **Stage 2 Corpus (Informational)** | 404 | 6 | 8.99 | 1096 |

---

## 📈 Model Performance Results

The classification performance of the trained models evaluated on the validation splits (20%) is summarized below:

### Stage 1 Classifier (Binary Filter)
*   **Accuracy:** `75.00%`
*   **AUC Score:** `0.8636`

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Non-Informational | 0.5714 | 0.8000 | 0.6667 | 10 |
| Informational | 0.8889 | 0.7273 | 0.8000 | 22 |

### Stage 2 Classifier (Multi-Class Category Classifier)
*   **Accuracy:** `70.37%`
*   **Weighted F1-score:** `70.81%`

| Information Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Community Support | 0.0000 | 0.0000 | 0.0000 | 2 |
| Damage Reports | 0.2105 | 1.0000 | 0.3478 | 4 |
| Evacuation & Safety | 0.0000 | 0.0000 | 0.0000 | 5 |
| Geolocation Info | 0.0000 | 0.0000 | 0.0000 | 3 |
| Official Announcements | 0.0000 | 0.0000 | 0.0000 | 1 |
| Weather Updates | 0.8983 | 0.8030 | 0.8480 | 66 |

*Note: The severe class imbalance in Stage 2 represents an area of active refinement. Future versions will incorporate synthetic data augmentation (using paraphrasing and back-translation) to balance minor categories.*

---

## 🔒 License & Ethics
*   **Ethics:** This research uses exclusively public, anonymized Reddit data. All personally identifiable information (PII) including usernames has been removed to preserve privacy.
*   **Usage:** Adheres to Reddit’s API Developer Terms.
