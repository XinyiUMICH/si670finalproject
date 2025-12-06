## **Project Pipeline: Data Preparation → Model Training → Visualization**

This project builds and evaluates search-relevance prediction models using TF-IDF features, Random Forest, and Linear Regression. The full workflow consists of three steps:

---

### **1. Data Preparation**

Run `prepare_data.py` to load raw data, clean text fields, build TF-IDF features, and generate Train/Valid/Test splits.

```bash
python prepare_data.py
```

This script creates a directory:

```
processed_data/
```

containing:

* TF-IDF sparse matrices
* label arrays
* question-group files
* baseline relevance scores

---

### **2. Model Training & Evaluation**

Run `train_and_evaluate_v2.py` using:

```bash
python train_and_evaluate_v2.py \
  --data_dir /Users/xujiaxuan/Desktop/si670finalproject-main/processed_data \
  --output_dir /Users/xujiaxuan/Desktop/si670finalproject-main/outputs \
  --rf_tune \
  --rf_n_iter 20 \
  --rf_cv_folds 3 \
  --cv_folds 5 \
  --top_k_errors 20
```

### **Arguments You May Modify**

| Argument         | Meaning                              | When to Change                                 |
| ---------------- | ------------------------------------ | ---------------------------------------------- |
| `--data_dir`     | Path to processed TF-IDF files       | If processed_data is in a different location   |
| `--output_dir`   | Where models & results will be saved | If you want a new output folder                |
| `--rf_tune`      | Enable Random Forest tuning          | Optional but recommended                       |
| `--rf_n_iter`    | # of random hyperparameter samples   | Increase for better tuning; decrease for speed |
| `--rf_cv_folds`  | CV folds during RF search            | 3–5 recommended                                |
| `--cv_folds`     | CV on final RF & LR models           | Default 5                                      |
| `--top_k_errors` | Export worst-K error cases           | Adjust the number of samples analyzed          |

### **Outputs Created**

Saved under `output_dir/`:

* `rf_model.joblib` — trained Random Forest
* `linreg_model.joblib` — trained Linear Regression
* `evaluation_results.json` — regression & ranking metrics
* `errors_all.csv` — per-sample prediction table
* `top_errors_rf.csv` / `top_errors_lr.csv` — worst-case errors

---

### **3. Visualization**

Open and run **`visualization.ipynb`** to generate plots such as:

* RMSE & R² comparison
* nDCG@5 & MRR comparison
* Scatter plots (true vs. predicted)
* Error distributions
* Other model analysis visuals

This notebook reads outputs from `outputs/` and produces figures for analysis and reporting.

---

### Summary of Complete Workflow

1. **Prepare data**

   ```
   python prepare_data.py
   ```

2. **Train and evaluate models**
   *(modify arguments as needed)*

   ```
   python train_and_evaluate_v2.py ...
   ```

3. **Visualize results**

   ```
   Open visualization.ipynb in Jupyter Notebook
   Run all cells
   ```


