from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb



def _encode_dates(X): # features engineering there
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "month_sin"] = np.sin(X["date"].dt.month * 2 * np.pi / 12)
    X.loc[:, "month_cos"] = np.cos(X["date"].dt.month * 2 * np.pi / 12)
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "weekday_sin"] = np.sin(X["date"].dt.weekday * 2 * np.pi / 7)
    X.loc[:, "weekday_cos"] = np.cos(X["date"].dt.weekday * 2 * np.pi / 7)
    X.loc[:, "hour"] = X["date"].dt.hour
    X.loc[:, "hour_sin"] = np.sin(X["date"].dt.hour * 2 * np.pi / 24)
    X.loc[:, "hour_cos"] = np.cos(X["date"].dt.hour * 2 * np.pi / 24)


    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date', 't', 'u', 'rr1', 'n', 'lockdown', 'ferie', 'curfew', 'vacances', 'rush_hour']].sort_values('date'), on='date')  # and all the things you want to add after the 't' and then yeah
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X



def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month_sin', 'day', 'weekday_sin', 'hour_sin', 'month_cos', 'weekday_cos', 'hour_cos']

    categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    categorical_cols = ["counter_name", "site_name"]

    numeric_cols = ['t', 'u', 'rr1', 'n', 'lockdown', 'ferie', 'curfew', 'vacances', 'rush_hour' ]

    preprocessor = ColumnTransformer([
        ('date', "passthrough", date_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('numeric', 'passthrough', numeric_cols)
    ])

    regressor = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=900, max_depth=8, min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            n_jobs=4, random_state=42)

    pipe =  make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor
    )

    return pipe
