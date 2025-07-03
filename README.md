# 🧼 Data Cleaning Summary

This repository documents the key steps and decisions made in cleaning the dataset composed of two primary CSV files: `True.csv` and `Fake.csv`.

---

## 📊 Dataset Overview

| File Name  | Original Rows | Notes                                  |
|------------|----------------|----------------------------------------|
| `True.csv` | ~21,211        | Generally clean with minimal missing values |
| `Fake.csv` | ~23,502        | Contains more missing and malformed data     |

---

## 🧹 Cleaning Strategy

### 🔄 Unified Cleaning Rules

Both datasets are cleaned using the **same criteria** for consistency:

- Rows are removed if **any** of the following fields are missing:
  - `title`
  - `text`
  - `subject`
  - `date`

> These four fields are essential for analysis. Missing data in any of them may skew results or reduce interpretability.

---

### ✅ `True.csv`

- **Cleaning Rule:** Drop rows with missing `title`, `text`, `subject`, or `date`
- **Observation:** Fewer rows dropped due to overall higher data quality
- **Outcome:** Majority of data retained for analysis

---

### ❌ `Fake.csv`

- **Cleaning Rule:** Drop rows with missing `title`, `text`, `subject`, or `date`
- **Observation:** More rows dropped due to frequent missing or corrupted values
- **Outcome:** Smaller but cleaner dataset ready for comparison

---

## 🔍 Post-Cleaning Sampling

To reduce processing time and support balanced analysis, a **30% stratified sampling** was applied to both cleaned datasets.

| Dataset     | Approx. Cleaned Rows | 30% Sample Size |
|-------------|-----------------------|------------------|
| `True.csv`  | ~20,000               | ~6,000 rows      |
| `Fake.csv`  | ~7,000                | ~2,100 rows      |

---

## 🔧 Mojibake Fix (Character Encoding Cleanup)

Some entries, mostly from `Fake.csv`, contained **mojibake** (corrupted characters) due to encoding issues. These were fixed using formulas in **Excel or Google Sheets**.

### Example Formula:

```excel
=SUBSTITUTE(
  SUBSTITUTE(
    SUBSTITUTE(
      SUBSTITUTE(
        SUBSTITUTE(
          SUBSTITUTE(
            SUBSTITUTE(
              SUBSTITUTE(
                SUBSTITUTE(
                  SUBSTITUTE(
                    SUBSTITUTE(
                      SUBSTITUTE(
                        A2,
                        "√¢¬Ä¬ú", "“"),
                      "√¢¬Ä¬ù", "”"),
                    "√¢¬Ä¬ôs", "’s"),
                  "√¢¬Ä¬ï", "’"),
                "√¢¬Ä¬î", "‘"),
              "√¢¬Ä¬ït", "’t"),
            "√¢¬Ä¬í", "“"),
          "√¢¬Ä¬ì", "”"),
        "√¢¬Ä¬¶", "…"),
      "√¢¬Ä¬ä", "—"),
    "√¢¬Ä¬", ""),
  "√¢", "")
