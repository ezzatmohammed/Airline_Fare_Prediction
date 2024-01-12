from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
import pandas as pd
import numpy as np


df=pd.read_csv("ml_df.csv")









X=df.drop('Price',axis=1)
y=df["Price"]



categorical_columns = ["Airline", "Source", "Destination"]
preprocessor = ColumnTransformer(
    transformers=[
        ("OHE",OneHotEncoder(sparse=False,drop = "first"), categorical_columns)
    ]
    ,remainder='passthrough'
)


pipeline = Pipeline([
    ("transform", preprocessor),
    ("scaler", MinMaxScaler()),
    ('model', XGBRegressor(learning_rate= 0.1, max_depth= 6, n_estimators= 200, reg_alpha= 1,
                           reg_lambda= 1, subsample= 0.9) )
])
pipeline.fit(X,y)


import pickle


pickle.dump(pipeline, open('model.pkl', 'wb'))
# Save the column names instead of the entire X DataFrame
with open('input_columns.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)