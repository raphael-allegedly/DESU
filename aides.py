import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency


'''  pour verifier la présence d'une colonne dans le dans le dataframe '''
def check_col(df, column_name):
    if column_name in df.columns:
        print(f" '{column_name}' est une variable du dataframe.")
    else:
        print(f" '{column_name}' n'est pas dans le dataframe.")
    

def string_to_list(input_string):
    # Split the string by commas, strip any leading/trailing spaces from each item
    return [item.strip() for item in input_string.split(',')]


def get_unique_values(df, columns):
    # Dictionary to store the results
    unique_values = {}
    
    # Iterate through the list of columns and get unique values for each
    for col in columns:
        unique_values[col] = df[col].unique().tolist()
    
    return unique_values

class FeatureValueMapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical1, categorical2, ordinal1, ordinal2, ordinal3, numerique1, numerique2):
        self.categorical1 = categorical1
        self.categorical2 = categorical2
        self.ordinal1 = ordinal1
        self.ordinal2 = ordinal2
        self.ordinal3 = ordinal3
        self.numerique1 = numerique1
        self.numerique2 = numerique2
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Make a copy of the dataframe to avoid changing the original
        X_transformed = X.copy()

        # Transformation for categorical1 (process column by column)
        cat1_values_to_nan = [9, 99, 999, 9999, -4]
        cat1_values_to_zero = [8, 88, 888, 8888]
        print("Before transformation of categorical1:")
        print(X_transformed[self.categorical1].head())
        for col in self.categorical1:
            X_transformed[col] = X_transformed[col].replace(cat1_values_to_nan, np.nan)
            X_transformed[col] = X_transformed[col].replace(cat1_values_to_zero, 0)
        print("After transformation of categorical1:")
        print(X_transformed[self.categorical1].head())

        # Transformation for categorical2 (process column by column)
        cat2_values_to_nan = [9, 99, 999, 9999, -4]
        print("Before transformation of categorical2:")
        print(X_transformed[self.categorical2].head())
        for col in self.categorical2:
            X_transformed[col] = X_transformed[col].replace(cat2_values_to_nan, np.nan)
        print("After transformation of categorical2:")
        print(X_transformed[self.categorical2].head())

        # Transformation for ordinal1 (process column by column)
        ord1_values_to_nan = [9, 99, 999, 9999, 8, 88, 888, 8888, -4]
        print("Before transformation of ordinal1:")
        print(X_transformed[self.ordinal1].head())
        for col in self.ordinal1:
            X_transformed[col] = X_transformed[col].replace(ord1_values_to_nan, np.nan)
        print("After transformation of ordinal1:")
        print(X_transformed[self.ordinal1].head())

        # Transformation for ordinal2 (process column by column)
        ord2_values_to_nan = [88, -4]
        print("Before transformation of ordinal2:")
        print(X_transformed[self.ordinal2].head())
        for col in self.ordinal2:
            X_transformed[col] = X_transformed[col].replace(ord2_values_to_nan, np.nan)
        print("After transformation of ordinal2:")
        print(X_transformed[self.ordinal2].head())

        # Transformation for ordinal3 (process column by column)
        ord3_values_to_nan = [9, -4]
        print("Before transformation of ordinal3:")
        print(X_transformed[self.ordinal3].head())
        for col in self.ordinal3:
            X_transformed[col] = X_transformed[col].replace(8, 0)
            X_transformed[col] = X_transformed[col].replace(ord3_values_to_nan, np.nan)
        print("After transformation of ordinal3:")
        print(X_transformed[self.ordinal3].head())

        # Transformation for numerique1 (process column by column)
        num1_values_to_nan = [99, 888, 88.8, 888.8, -4]
        print("Before transformation of numerique1:")
        print(X_transformed[self.numerique1].head())
        for col in self.numerique1:
            X_transformed[col] = X_transformed[col].replace(num1_values_to_nan, np.nan)
        print("After transformation of numerique1:")
        print(X_transformed[self.numerique1].head())

        # Transformation for numerique2 (process column by column)
        num2_values_to_nan = [8888, 88, 8, 9, 99, 999, 9999, -4, 777]
        print("Before transformation of numerique2:")
        print(X_transformed[self.numerique2].head())
        for col in self.numerique2:
            X_transformed[col] = X_transformed[col].replace(num2_values_to_nan, np.nan)
        print("After transformation of numerique2:")
        print(X_transformed[self.numerique2].head())

        return X_transformed
    
    
def cramers_v(x, y):
    """Calculates Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# Function to calculate correlation matrix using Cramér's V
def cramers_v(x, y):
    """Calculates Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# Function to calculate correlation matrix using Cramér's V
def cramers_v_matrix(df, categorical_features, threshold=0.8):
    """Calculates the Cramér's V matrix for the provided categorical features and detects highly correlated pairs."""
    # Initialize an empty set for columns to remove due to high correlation
    cols_to_remove = set()

    # Initialize an empty DataFrame to store Cramér's V values
    cramers_v_df = pd.DataFrame(index=categorical_features, columns=categorical_features)
    
    for i in range(len(categorical_features)):
        for j in range(i, len(categorical_features)):
            if i == j:
                cramers_v_df.iloc[i, j] = 1.0  # Diagonal elements are 1
            else:
                v = cramers_v(df[categorical_features[i]], df[categorical_features[j]])
                cramers_v_df.iloc[i, j] = cramers_v_df.iloc[j, i] = v
                
                # If Cramér's V is above the threshold, mark one of the columns to be removed
                if v >= threshold:
                    cols_to_remove.add(categorical_features[j])  # Arbitrarily remove the second feature in the pair
                    


    return cramers_v_df, cols_to_remove