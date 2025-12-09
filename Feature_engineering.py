import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce
from lightgbm import LGBMRegressor
import joblib



# In[40]:


#1. Load the cleaned database
clean_database = pd.read_csv("Cleaned_database.csv")

#2. Split the data in Features and Target
X_features = clean_database.drop('Price', axis=1)
Y = clean_database['Price']

#3. Split data in train and test sets
x_train_0,x_test_0,y_train,y_test = train_test_split(X_features,Y,train_size=.8,random_state=42)




# In[41]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce 


# features_selected = ['Year','Kilometers','Gearbox','Brand','Model','Variant']
# 1. Define column groups
cat_ohe_cols = ['Gearbox'] 
cat_ordinal_cols = []      
cat_target_cols = ['Brand', 'Model', 'Variant'] 
num_cols = ['Year', 'Kilometers']




# Pipeline for numerical: Impute missing -> Scale
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

# Pipeline for OneHot: Impute -> Encode
ohe_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Pipeline for Target Encoding: Impute -> Encode
# TargetEncoder handles new categories in test data automatically
target_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', ce.TargetEncoder()),
    ('scaler',MinMaxScaler()) 
])

# Pipeline for Ordinal Encoding: Impute -> Encode
order = ['New In Stock','Demo','Used']
ordinal_encoder = OrdinalEncoder(categories=[order])
ordinal_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', ordinal_encoder),
    ('scaler',MinMaxScaler())
])

# 3. Combine into a preprocessor
preprocessor_final = ColumnTransformer(transformers=[
    ('num', num_pipe, num_cols),
    ('ohe', ohe_pipe, cat_ohe_cols),
    ('target', target_pipe, cat_target_cols),
    ('Ordinal', ordinal_pipe, cat_ordinal_cols)
], remainder='drop')

# 4. Apply
# Fit only on TRAIN, transform TRAIN AND TEST

x_train_final = preprocessor_final.fit_transform(x_train_0, y_train)
x_test_final = preprocessor_final.transform(x_test_0)


feature_names = preprocessor_final.get_feature_names_out()
#Calling the model from .ipynb file. Is that neccesary?

