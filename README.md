# Laptop-Price-Detection-Model-
Predicting gaming laptop prices in Malaysian Ringgit using Data Engineering and Random Forest regression.

💻 Gaming Laptop Price Predictor (MYR)

An end-to-end Machine Learning project that predicts the price of gaming laptops in Malaysian Ringgit (MYR) based on complex hardware specifications. 

Project Overview
This project processes raw e-commerce data for gaming laptops, engineers high-value hardware features from unstructured text, normalizes price distributions, and trains a highly tuned Random Forest model to predict retail prices.

**Advanced Features Implemented:**
- **Deep Feature Engineering:** Developed custom Regex functions to extract CPU architectures, GPU Tiers, RAM capacity, and Storage size from messy product titles.
- **Log Transformation (`np.log1p`):** Applied logarithmic transformations to the target variable to handle right-skewed pricing distributions and minimize outlier impact, reversing it (`np.expm1`) for final evaluation.
- **Currency Localization:** Automated USD to MYR conversion for localized market analysis.
- **Hyperparameter Optimization:** Utilized `GridSearchCV` with multi-threading (`n_jobs=-1`) to automatically discover the optimal tree depth, sample splits, and estimator count.

## 🛠️ Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-Learn
- **Algorithm:** Random Forest Regressor

## 📊 Dataset Attribution
The raw data used in this project was sourced from Kaggle: [Gaming Laptops 2026 Dataset](https://www.kaggle.com/datasets/kanchana1990/gaming-laptops-2026). 
*Note: To respect the author's licensing, the raw dataset is not included in this repository. You can download it directly from Kaggle.*

## 👨‍💻 Author
**Agatteshan**
