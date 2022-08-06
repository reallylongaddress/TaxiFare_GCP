import os
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from TaxiFare_GCP import gcp_params
from google.cloud import storage

PATH_TO_LOCAL_MODEL = 'model.joblib'
AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"
LOCAL_GCP_MODEL_PATH = './models/gcp_model_2.joblib'

def get_test_data(data="gcp", nrows=1000):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"  # ⚠️ to test from actual KAGGLE test set for submission

    if data == "local":
        df = pd.read_csv(path)
    elif data == "full":
        df = pd.read_csv(AWS_BUCKET_TEST_PATH)
    elif data == "gcp":
        print(f'get_test_data GCP: gs://{gcp_params.BUCKET_NAME}/{gcp_params.BUCKET_TEST_DATA_PATH}')
        df = pd.read_csv(f"gs://{gcp_params.BUCKET_NAME}/{gcp_params.BUCKET_TEST_DATA_PATH}")
    else:
        df = pd.read_csv(AWS_BUCKET_TEST_PATH, nrows=nrows)
    return df


def get_model(source='local'):
    print(f'get_model: {source}')
    if source == 'local':
        pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
        return pipeline
    elif source == 'gcp':

        print(f'get_model GCP:')
        client = storage.Client()
        bucket = client.bucket(gcp_params.BUCKET_NAME)
        print(f'get_model GCP: {bucket}')
        blob = bucket.blob(f'{gcp_params.STORAGE_LOCATION}{gcp_params.MODEL_NAME}')
        print(f'downloading LOCAL_GCP_MODEL_PATH: {blob}')

        blob.download_to_filename(LOCAL_GCP_MODEL_PATH)
        print(f'downloaded LOCAL_GCP_MODEL_PATH: {LOCAL_GCP_MODEL_PATH}')
        pipeline = joblib.load(LOCAL_GCP_MODEL_PATH)
        return pipeline
    else:
        raise Exception("Unknown model source path")



def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(nrows, kaggle_upload=False):
    df_test = get_test_data(data='gcp')

    pipeline = get_model(source='gcp')
    print(f'pipeline: {type(pipeline)}')

    if "best_estimator_" in dir(pipeline):
        print("pipeline A")
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
        print("pipeline B")
    print(f'y_pred:  {y_pred.shape}/{df_test.shape}')

    # pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    # if "best_estimator_" in dir(pipeline):
    #     y_pred = pipeline.best_estimator_.predict(df_test)
    # else:
    #     y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # # Set kaggle_upload to False unless you install kaggle cli
    # if kaggle_upload:
    #     kaggle_message_submission = name[:-4]
    #     command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
    #     os.system(command)


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    nrows = 100
    generate_submission_csv(nrows, kaggle_upload=False)
