import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
data = pd.read_csv('train.csv')

# One-hot encoding for nominal categorical variables
nominal_categories = ['MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'LandSlope',
                      'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
                      'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 
                      'Heating', 'CentralAir', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition', 'MiscFeature']
# Apply label encoding to each categorical feature
for feature in nominal_categories:
    data[feature] = data[feature].astype('category').cat.codes

# Label encoding for ordinal categorical variables
ordinal_categories = {
    'Alley': {'NA': 0, 'Grvl': 1, 'Pave': 2},
    'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
    'Utilities': {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3},
    'ExterQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtQual': {'Po': 0, 'NA': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'Po': 0, 'NA': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'KitchenQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
    'FireplaceQu': {'Po': 0, 'NA': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
    'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
}

for col, mappings in ordinal_categories.items():
    data[col] = data[col].map(mappings).fillna(0)

# Creating polynomial features
# data['LotArea2'] = data['LotArea'] ** 2
# data['TotalBsmtSF2'] = data['TotalBsmtSF'] ** 2
# data['GrLivArea2'] = data['GrLivArea'] ** 2
# data['MasVnrArea2'] = data['MasVnrArea'] ** 2
# data['GarageArea2'] = data['GarageArea'] ** 2

# # Interaction features
# data['TotalSF'] = data['TotalBsmtSF'] + data['FirstFlrSF'] + data['SecondFlrSF']
# data['TotalBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])
# data['LivingAreaQuality'] = data['GrLivArea'] * data['OverallQual']
# data['TotalOutdoorArea'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['Threeseasonporch'] + data['ScreenPorch'] + data['PoolArea']
# data['OverallScore'] = data['OverallQual'] * data['OverallCond']
# data['TotalArea'] = data['GrLivArea'] + data['TotalBsmtSF'] + data['GarageArea']
# data['OverallQualCond'] = data['OverallQual'] * data['OverallCond']

# # Ratios between features
# epsilon = 1
# data['BedBathRatio'] = data['BedroomAbvGr'] / (data['FullBath'] + 0.5 * data['HalfBath'] + epsilon)
# data['LotFrontageToLotAreaRatio'] = data['LotFrontage'] / (data['LotArea'] + epsilon)
# data['LivingAreaToTotalAreaRatio'] = data['GrLivArea'] / (data['TotalArea'] + epsilon)

# # Transformed Features
# data['LogTotalArea'] = np.log(data['TotalArea'] + epsilon)
# Apply square root or exponential transformations to other features as needed

# Composite Scores
# Assuming 'Exterior1st' and 'Exterior2nd' are encoded in a way that can be combined
# data['ExteriorComposite'] = data['Exterior1st'] + data['Exterior2nd']
# # For BasementScore, you need to ensure 'BsmtQual', 'BsmtCond', and 'BsmtFinType1' are numeric
# data['BasementScore'] = data['BsmtQual'] + data['BsmtCond'] + data['BsmtFinType1']


# # Age and Remodeling Features
# data['AgeAtSale'] = data['YrSold'] - data['YearBuilt']
# data['YearsSinceRemodel'] = data['YrSold'] - data['YearRemodAdd']
# data['AgeAtSaleTotalArea'] = data['AgeAtSale'] * data['TotalArea']

# # Scale the polynomial features
# scaler = MinMaxScaler()
# # polynomial_features = ['LotArea2', 'TotalBsmtSF2', 'GrLivArea2', 'MasVnrArea2', 'GarageArea2', 'TotalArea', 'OverallScore', 'TotalOutdoorArea', 'LivingAreaQuality', 'BedBathRatio']
# polynomial_features = ['LotArea2']
# data[polynomial_features] = scaler.fit_transform(data[polynomial_features])

# Check for NaN in numerical features and replace with 0
numerical_columns = data.select_dtypes(include=[np.number]).columns
data[numerical_columns] = data[numerical_columns].fillna(0)

# Drop original columns that were transformed
# data.drop(['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'FullBath', 'HalfBath', 
#            'BsmtFullBath', 'BsmtHalfBath', 'YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)

#drop the features with low or small mutual information score
# Drop original columns that were transformed
# data.drop(['MoSold', 'PoolArea', 'Threeseasonporch', 'Functional', 'LowQualFinSF', 'Utilities', 'KitchenAbvGr', 'Street', 'MiscVal', 'LandSlope', 'YrSold', 'MiscFeature', 'PoolQC'], axis=1, inplace=True)

# Save the transformed dataset
data.to_csv('train5.csv', index=False)
