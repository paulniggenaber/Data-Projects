# Football Transfer Fee Prediction using Machine Learning

## Brief Summary
This project develops and evaluates machine learning models to predict professional football transfer fees using player performance, contract, league, and contextual features. The focus lies on careful data preprocessing model validation.
The most extensive description can be found in final_report.pdf.
---

## Project Overview

### Motivation
Transfer fees in professional football are influenced by both objective performance indicators and subjective expectations about a player’s future value. Accurately modeling these fees is challenging due to heavy skewness, missing data, and complex nonlinear relationships. This project aims to quantify these effects using modern machine learning techniques while maintaining interpretability and methodological rigor.

---

### Data Description
The dataset combines multiple sources of player- and transfer-related information, including:

- **Transfer data**
  - Transfer fee (target variable)
  - Transfer window and season
  - Selling and buying leagues
  - Free transfers excluded

- **Player characteristics**
  - Age, position, height
  - Market and performance indicators
  - Contract-related variables

- **Derived features**
  - Log-transformed transfer fees
  - Aggregated performance metrics

Missing values are handled systematically, primarily via **median imputation**, which is robust to skewed distributions typical for football data. Only features relevant for prediction are retained; non-informative identifiers (e.g. shirt numbers) are removed.

---

### Exploratory Data Analysis
The exploratory analysis focuses on:

- Distributional properties of transfer fees (raw and log-transformed)
- Differences across leagues
- Visulaization of non-linear relationship
- Correlation structures among predictors
- Identification and removal of highly collinear features (absolute correlation > 0.9)

These steps ensure numerical stability and reduce redundancy before model training.

---

### Modeling Approach
The core predictive model is **XGBoost** for both classification and regression and CatBoost for regression. Tuning was done by RnadomizedSearchCV.

Key modeling elements include:
- Log-transformed target variable to address skewness
- Train–validation split for performance evaluation
- Hyperparameter tuning with emphasis on regularization
- Explicit comparison of L1 and L2 penalties

Model performance is evaluated using:
- Root Mean Squared Error (RMSE)
- Out-of-sample \( R^2 \)
- Learning curves across boosting rounds

---

### Regularization Analysis
A central component of the project is the analysis of regularization effects in XGBoost:

- Different combinations of L1 (`alpha`) and L2 (`lambda`) penalties are compared
- Train vs. validation performance is tracked over boosting rounds
- Overfitting behavior is explicitly analyzed and visualized

This allows for transparent model selection and guards against overly optimistic in-sample results.

---

### Results
The results demonstrate that:
- Log transformation substantially improves model stability
- Regularization plays a crucial role in preventing overfitting
- League and performance-related variables are among the strongest predictors
- Validation performance peaks well before the maximum number of boosting rounds

The final model balances predictive accuracy with generalizability.

---

### Limitations
- Transfer fees reflect negotiation dynamics not fully observable in the data
- Subjective expectations are only indirectly captured
- Results should be interpreted as predictive, not causal

---

### Future Extensions
- Incorporation of text-based features (e.g. news or scouting reports)
- Incorporation of fame based features (Instagram follower)

---
## Project Pipeline
transfermarketscraper + transfer_parser.py + tr_scraping.py --> transfer_dataset_4.csv, transfer_dataset_5.csv --> build_transfer_data_csv.py --> transfer_data.csv

fbrefscraper.py + build_player_data_csv.py --> player_data.csv

player_data.csv + transfer_data.csv + run.py --> dataprocessor.py + models.py --> results
