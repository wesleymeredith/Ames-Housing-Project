# Ames Housing Project

The Ames Housing Project employs advanced machine learning techniques, including XGBoost, Artificial Neural Network (ANN), Random Forest, and Decision Tree algorithms, to predict housing prices. This project provides a comprehensive set of tools to clean and analyze housing data, making it a good resource for real estate professionals or data enthusiasts.

## Usage

## Dependencies
Ensure you have the following dependencies installed:
- Sklearn
- Matplotlib
- GraphVIZ
- Pandas
- NumPy
  
### Data Cleaning
To clean the data, use the `cleanData.py` script. Run it twice, once with `test.csv` and once with `train.csv`. Replace the internal variable in the file accordingly.

```bash
python cleanData.py --input_file test.csv
python cleanData.py --input_file train.csv
```

## Feature Engineering
To perform feature engineering on the data, utilize the FeatureE.py script. Run it twice, once with test.csv and once with train.csv. Replace the internal variable in the file accordingly.
```bash
python FeatureE.py --input_file test.csv
python FeatureE.py --input_file train.csv
```


## Data Files
Use train4.csv and test4.csv for training, as these files contain the original data transformed and cleaned.


## Source
The dataset is from the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) Kaggle competition.

