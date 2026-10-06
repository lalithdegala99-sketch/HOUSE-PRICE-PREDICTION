# House Price Prediction

Machine learning project that predicts house prices from area, bedrooms, bathrooms, age and location score.

> Note: this version uses a sample (synthetic) dataset so the code runs anywhere. To use real data, replace the DATA section in the script with `pd.read_csv("your_file.csv")`.

## Tools Used
Python, pandas, numpy, scikit-learn, matplotlib, seaborn

## Models
- Linear Regression (baseline)
- Random Forest Regressor

## Results (sample dataset)
| Model | R2 | MAE (lakhs) |
|---|---|---|
| Linear Regression | 0.971 | 6.11 |
| Random Forest | 0.954 | 7.80 |

## Charts
![Actual vs Predicted](2_actual_vs_predicted.png)
![Feature Importance](3_feature_importance.png)
![Correlation Heatmap](1_correlation_heatmap.png)
![Model Comparison](4_model_comparison.png)

## How to Run
```
pip install -r requirements.txt
python house_price_prediction.py
```

## Author
Lalith Kumar Degala
