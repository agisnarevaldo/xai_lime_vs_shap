# Project Analysis Report: xai_lime_vs_shap

## 1. Project Overview
The **xai_lime_vs_shap** project is designed to compare two prominent Explainable AI (XAI) frameworks—**LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (SHapley Additive exPlanations). The target application appears to be interpreting machine learning models trained on **Tokopedia product reviews**.

## 2. Technical Stack
- **Languages**: Python (Scripts & Jupyter Notebooks)
- **Scraping**: Playwright, playwright-stealth, BeautifulSoup/Selector logic.
- **Data Handling**: Pandas, JSON, CSV.
- **Workflow**: 
    - Scraping -> Cleaning -> Modeling -> Explainability.

## 3. Project Structure
```text
xai_lime_vs_shap/
├── data/
│   ├── raw/           # Multiple versions of scraped Tokopedia data (JSON/CSV)
│   └── processed/     # Cleaned data ready for modeling
├── notebooks/         # Core research pipeline
│   ├── 01_scraping.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── src/               # Reusable code (models, preprocessing, scraping)
├── scripts/           # Utility scripts (located in root: merge, check, analyze)
└── .github/prompts/   # Task planning and active prompts
```

## 4. Current State Analysis
### Data Collection
The project has a robust scraping infrastructure. Given the variety of scripts (`scrape_low_ratings.py`, `scrape_additional.py`), there is a specific focus on handling imbalanced data or targeted review types (e.g., low ratings).

- **Current focus**: Scraping **iPhone 17** data from Tokopedia (per `plan-scrapingIphone17Tokopedia.prompt.md`).
- **Data Quality**: Tools like `analyze_reviews.py` are used to monitor extraction rates and identify failed scrape attempts.

### Modeling & Explainability
While the scraping phase is currently active, the infrastructure for modeling and XAI comparison is already in place via the `notebooks` directory. This suggests the project is in a cyclical refinement phase—gathering more targeted data to improve or further test the XAI comparisons.

## 5. Key Challenges & Strategies
- **Anti-Bot Protection**: The project uses `playwright-stealth` and favors "headful" (non-headless) browsing to bypass Tokopedia's detection.
- **Data Merging**: Extensive logic exists for merging new crawls with existing backups while maintaining data integrity.

## 6. Recommendations / Observation
- **README**: The main `README.md` is currently empty. It would benefit from a high-level description of the XAI research goals.
- **Consistency**: The root directory is somewhat cluttered with various utility scripts. These could be moved to a `scripts/` or `tools/` directory to match the clean structure of `notebooks/` and `src/`.
- **Environment**: Ensure a `requirements.txt` or `environment.yml` is maintained, as the stack (Playwright, XAI libs) can be sensitive to versioning.

---
*Report generated on April 13, 2026.*
