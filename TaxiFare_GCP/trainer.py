import pandas as pd
import joblib
from termcolor import colored
import mlflow

from TaxiFare_GCP import gcp_params
from TaxiFare_GCP.data import get_data, clean_data, get_test_data
from TaxiFare_GCP.encoders import TimeFeaturesEncoder, DistanceTransformer, NumericOptimizer
from TaxiFare_GCP.utils import compute_rmse

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from google.cloud import storage

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[REMOTE] [reallylongaddress] TaxiFareModel_GCP + 0.0.9"
#GCP_MODEL_NAME='gcp_model_2.joblib'

class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self, estimator='LinearRegression'):
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

        # self.pipeline = Pipeline([
        #     ('preproc', preproc_pipe),
        #     ('numeric_optimizer', NumericOptimizer()),
        #     ('linear_model', LinearRegression())
        # ])
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('numeric_optimizer', NumericOptimizer()),
            ('knn_model', KNeighborsRegressor(n_neighbors=10))
        ])

    def run(self):

        self.set_pipeline()
        self.mlflow_log_param("model", "knn")
        self.mlflow_log_metric("train_val rows", f'{len(X)}')
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.mlflow_log_metric("test rows", f'{len(X_test)}')
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse, 2)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    # def save_model(self):
    #     """Save the model into a .joblib format"""
    #     joblib.dump(self.pipeline, 'model.joblib')
    #     print(colored("model.joblib saved locally", "green"))

    def save_model(self, reg):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(reg, gcp_params.MODEL_NAME)
        print(f"saved {gcp_params.MODEL_NAME} locally")

        # Implement here
        self.upload_model_to_gcp()
        print(f"uploaded {gcp_params.MODEL_NAME} to gcp cloud storage under \n => {gcp_params.STORAGE_LOCATION}/{gcp_params.MODEL_NAME}")

    def upload_model_to_gcp(self):

        client = storage.Client()
        bucket = client.bucket(gcp_params.BUCKET_NAME)
        blob = bucket.blob(f'{gcp_params.STORAGE_LOCATION}{gcp_params.MODEL_NAME}')

        blob.upload_from_filename(gcp_params.MODEL_NAME)

    def save_submission(self, y_pred, estimator='KNN'):

        y_pred = pd.concat([df_test["key"],pd.Series(y_pred)],axis=1)
        y_pred.columns = ['key', 'fare_amount']
        pd.DataFrame(y_pred).to_csv(f'./submission_{estimator}.csv', index=False)

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        print('mlflow_client')
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        print('mlflow_experiment_id')
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            id = self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id
            print(f'mlflow_experiment_id: {id}')
            return id

    @memoized_property
    def mlflow_run(self):
        print('mlflow_run')
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        print(f'mlflow_log_param: {key}:{value}')
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        print(f'mlflow_log_metric: {key}:{value}')
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # Get and clean data
    N = 10_000
    df = get_data(nrows=N)

    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    print(f'X/y SHAPE: {X.shape}/{y.shape}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and

    print(f'Xtrain/y SHAPE: {X_train.shape}/{y_train.shape}')
    trainer = Trainer(X=X_train, y=y_train)
    trainer.set_experiment_name(EXPERIMENT_NAME)
    trainer.run()

    print(f'Xtest/y SHAPE: {X_test.shape}/{y_test.shape}')
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")

    df_test = get_test_data()
    y_pred = trainer.predict(df_test)

    # print(y_pred_list)
    trainer.save_submission(y_pred)
    trainer.save_model(trainer.pipeline)
