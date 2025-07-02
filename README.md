# 🧼 Data Cleaning Summary

This section documents the key decisions and status of the data cleaning process for our dataset composed of two primary CSV files: `True.csv` and `Fake.csv`.

---

## 📊 Dataset Overview

| File Name  | Original Rows | Notes                          |
|------------|----------------|-------------------------------|
| True.csv   | ~21,211        | Mostly clean; fewer missing values |
| Fake.csv   | ~23,502        | Contains more missing and corrupted data |

---

## 🧹 Cleaning Strategy

### ✅ `True.csv`
- **Cleaning Rule:** Remove rows **only where the `content` field is missing**.
- **Result:** Less aggressive cleaning, retains rows even if `title`, `subject`, or `date` are missing.
- **Outcome:** A high number of rows preserved for sampling and analysis.

### ❌ `Fake.csv`
- **Cleaning Rule:** Remove rows **where any column is missing** (`title`, `text`, `subject`, or `date`).
- **Result:** More aggressive cleaning, more rows removed due to frequent missing data.
- **Outcome:** Lower number of usable rows remaining after cleaning.

---

## 🔍 Post-Cleaning Sampling

To create a manageable and balanced dataset for dashboarding and analysis, **30% stratified sampling** was applied to both cleaned datasets.

### Example:
- `True.csv`: 20,000 cleaned rows → 30% sample → ~6,000 rows
- `Fake.csv`: 7,000 cleaned rows → 30% sample → ~2,100 rows

---

## 🔧 Mojibake Fix: Character Encoding Cleanup

Some rows from `Fake.csv` (and a few from `True.csv`) included **corrupted characters (mojibake)** due to encoding errors from the original source.

These were resolved using a formula-based approach in **Excel/Google Sheets**, such as the one below:

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
