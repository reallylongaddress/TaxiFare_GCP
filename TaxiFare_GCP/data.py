import pandas as pd
from TaxiFare_GCP import gcp_params

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"

def get_data(nrows=1000):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #todo: put read local if doing rapid local iterations
    df = pd.read_csv(f"gs://{gcp_params.BUCKET_NAME}/{gcp_params.BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    df = optimize_numierics(df)
    print(f'-----TRAIN getdata: {df.shape}')
    return df

def get_test_data():
    df = pd.read_csv(f"gs://{gcp_params.BUCKET_NAME}/{gcp_params.BUCKET_TEST_DATA_PATH}")
    df = optimize_numierics(df)
    print(f'-----TEST getdata: {df.shape}')
    return df

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
    return df

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
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("RAM Reduced by {} % | {} GB".format(ratio, GB))
    return df

if __name__ == '__main__':
    df = get_data()
