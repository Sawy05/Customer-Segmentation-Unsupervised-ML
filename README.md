# 📊 Customer Segmentation & Unsupervised Machine Learning Pipeline

An end-to-end **Customer Behavioral Segmentation & Retail Analytics** repository implementing three distinct unsupervised clustering paradigms—**Agglomerative Hierarchical Clustering**, **Partitioning Around Medoids (K-Medoids)**, and **Fuzzy C-Means (`skfuzzy`)**—alongside an interactive **Power BI Executive Retail Report**.

---

## 🧠 Clustering Methodologies Implemented (`customer_segmentation_engine.py`)

1. **Feature Engineering & Standardization**:
   - Computes composite customer metrics (`Total_Spending` across 6 product categories, `Age`, household `Children`, `Income`, and `Recency`) and normalizes via `StandardScaler`.
2. **Agglomerative Hierarchical Clustering (`scikit-learn`)**:
   - Bottom-up ward linkage tree partitioning customers into 4 behavioral tiers.
3. **K-Medoids Clustering (`pyclustering`)**:
   - Outlier-resilient Euclidean medoid optimization selecting real customer exemplars as cluster centers.
4. **Fuzzy C-Means Soft Clustering (`scikit-fuzzy`)**:
   - Computes continuous membership probabilities (`Fuzzy_0` … `Fuzzy_3`) for customers straddling multiple behavioral segments.

---

## 📂 Repository Contents

- `customer_segmentation_engine.py` — Core Python unsupervised learning pipeline (Hierarchical, K-Medoids, Fuzzy C-Means).
- `marketing_campaign.csv` — Input customer demographic and multi-channel spend dataset.
- `clustered_output.csv` — Generated dataset enriched with hard cluster labels and soft fuzzy membership probabilities.
- `Online_Retail_Analytics_Report.pdf` & `OnlineRetail_Dashboard.pbix` — Executive BI visualization report and Power BI dashboard.

---

## 🚀 Quick Start

```bash
pip install pandas numpy scikit-learn scikit-fuzzy pyclustering matplotlib
python customer_segmentation_engine.py
```
