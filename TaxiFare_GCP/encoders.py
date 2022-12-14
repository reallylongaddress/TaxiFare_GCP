import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin

from TaxiFare_GCP.utils import haversine_vectorized
# from TaxiFare_GCP.data import optimize_numierics

class NumericOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_optimized = optimize_numierics(pd.DataFrame(X_.todense()))
        X_optimized = scipy.sparse.csr_matrix(X_optimized.values)
        return X_optimized

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extract the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'
    """

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Compute the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(
            X_,
            start_lat=self.start_lat,
            start_lon=self.start_lon,
            end_lat=self.end_lat,
            end_lon=self.end_lon
        )
        return X_[['distance']]

def optimize_numierics(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    # out_size = df.memory_usage(index=True).sum()
    # ratio = (1 - round(out_size / in_size, 2)) * 100
    # GB = out_size / 1000000000
    # if verbose:
    #     print("RAM Reduced by {} % | {} GB".format(ratio, GB))
    return df
