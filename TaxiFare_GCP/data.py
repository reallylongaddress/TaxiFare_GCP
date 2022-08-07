import pandas as pd
from TaxiFare_GCP import gcp_params
from TaxiFare_GCP.encoders import TimeFeaturesEncoder, DistanceTransformer, NumericOptimizer, optimize_numierics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"



def train_val_get_raw_data(nrows=1000):
    print(f'train_val_get_raw_data: {nrows}')
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #todo: put read local if doing rapid local iterations
    df = pd.read_csv(f"gs://{gcp_params.BUCKET_NAME}/{gcp_params.BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    df = optimize_numierics(df)
    print(f'-----TRAIN getdata: {df.shape}')
    return df

def test_get_raw_data():
    df = pd.read_csv(f"gs://{gcp_params.BUCKET_NAME}/{gcp_params.BUCKET_TEST_DATA_PATH}")
    df = optimize_numierics(df)
    print(f'-----TEST getdata: {df.shape}')
    return df

def get_preprocessing_pipeline():

    """defines the pipeline as a class attribute"""
    dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
    ])
    time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, [
            "pickup_latitude",
            "pickup_longitude",
            'dropoff_latitude',
            'dropoff_longitude'
        ]),
        ('time', time_pipe, ['pickup_datetime'])
    ], remainder="drop")

    pipeline = Pipeline([
        ('preproc', preproc_pipe),
        ('numeric_optimizer', NumericOptimizer()),
        # ('knn_model', KNeighborsRegressor(n_neighbors=10))
    ])

    return pipeline

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]

    data_pipeline = get_preprocessing_pipeline()
    return df


if __name__ == '__main__':
    df = train_val_get_raw_data()
