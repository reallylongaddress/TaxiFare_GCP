# from msilib.schema import _Validation_records
# from sqlite3.dbapi2 import _Parameters
# from this import d
import pandas as pd
import joblib
from termcolor import colored
import mlflow
import time

import sys
import os
import platform

import io

import TaxiFare_GCP.data
from TaxiFare_GCP import gcp_params, utils
# from TaxiFare_GCP.data import get_preprocessing_pipeline, get_raw_data
#from TaxiFare_GCP.data import import get_raw_data, clean_data, get_test_data
# from TaxiFare_GCP.utils import compute_rmse

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from google.cloud import storage

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[REMOTE] [reallylongaddress] TaxiFareModel_GCP + 0.0.11"
#GCP_MODEL_NAME='gcp_model_2.joblib'
BEST_MODEL = None

class Trainer(object):
    def __init__(self, params):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        # self.estimator = estimator
        self.params = params
        # self.X = X
        # self.y = y
        # for MLFlow
        self.data_pipeline = None
        self.experiment_name = EXPERIMENT_NAME

        print('mlflow_experiment_id')
        mlflow.set_tracking_uri(MLFLOW_URI)
        ml_flow_client = MlflowClient()
        try:
            self.mlflow_experiment_id = ml_flow_client.create_experiment(self.experiment_name)
            print(f'mlflow_experiment_id a: {self.mlflow_experiment_id}')
        except BaseException:
            self.mlflow_experiment_id = ml_flow_client.get_experiment_by_name(self.experiment_name).experiment_id
            print(f'mlflow_experiment_id b: {self.mlflow_experiment_id}')

        print(f'::{self.mlflow_experiment_id}::{self.experiment_name}::')

    def run(self):


        process_start_time = int(round(time.time(), 0))

        # print('Trainer.run')
        self.data_pipeline = TaxiFare_GCP.data.get_preprocessing_pipeline()
        # print(f'type: {type(self.data_pipeline)}')
        df_train_val = TaxiFare_GCP.data.train_val_get_raw_data(self.params.get('nrows'))
        # print(f'A>>>>df_train_val>>{df_train_val.isna().sum().sum()}<<')
        # print(f'df_train_val.shape: {df_train_val.shape}')

        df_train_val.dropna(how='any', axis=0, inplace=True)
        # print(f'B>>>>df_train_val>>{df_train_val.isna().sum().sum()}<<')
        # print(f'df_train_val.shape: {df_train_val.shape}')

        X_train_val = df_train_val.drop(columns=['fare_amount'])
        y_train_val = df_train_val['fare_amount']

        X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, test_size=0.20)
        # print(f'>>>>X_train>>{X_train.isna().sum().sum()}<<')

        self.data_pipeline.fit(X_train)

        X_train_preprocessed = self.data_pipeline.transform(X_train)
        # print(f'>>>>X_train_preprocessed>>{pd.DataFrame.sparse.from_spmatrix(X_train_preprocessed).isna().sum().sum()}<<')

        X_val_preprocessed = self.data_pipeline.transform(X_validate)

        X_test = TaxiFare_GCP.data.test_get_raw_data()
        X_test_preprocessed = self.data_pipeline.transform(X_test)
        # print(f'>>>>X_test_preprocessed>>{pd.DataFrame.sparse.from_spmatrix(X_test_preprocessed).isna().sum().sum()}<<')

        for estimator_name, hyperparams in self.params.get('estimators').items():

            loop_start_time = round(time.time(), 0)
            print(f'key: {estimator_name}')

            ml_flow_client = MlflowClient()
            ml_flow_run = ml_flow_client.create_run(self.mlflow_experiment_id)

            print(f'mlflow_log_param: start_time:{process_start_time}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'starttime', f'{process_start_time}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'model', estimator_name)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'train_size', f'{X_train_preprocessed.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'validate_size', f'{X_val_preprocessed.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'test_size', f'{X_test_preprocessed.shape[0]}')

            ml_flow_client.log_param(ml_flow_run.info.run_id, 'os.name', os.name)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'platform.system', platform.system())
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'platform.release', platform.release())

            #dbd todo - this is ugly, reduce to 1 line
            for param_key, param_value in hyperparams.items():
                ml_flow_client.log_param(ml_flow_run.info.run_id, param_key, param_value)
            print(f'self.mlflow_experiment_id: {self.mlflow_experiment_id}')

            grid = None
            model = None
            if estimator_name == 'knn':
                model = KNeighborsRegressor()

            elif estimator_name == 'sgd':
                model = SGDRegressor()

            elif estimator_name == 'linear':
                model = LinearRegression()
            else:
                raise Exception("Unknown model type")

            print(f'===={estimator_name}: {len(hyperparams)}')
            print(f'type: {type(hyperparams)}')
            print(f'----{hyperparams.get("hyperparams")}')
            grid = GridSearchCV(model,
                                param_grid=hyperparams.get("hyperparams"),
                                cv=3,
                                scoring='neg_root_mean_squared_error',
                                # return_train_score=True,
                                verbose=1,
                                n_jobs=-1
                                )

            grid.fit(X_train_preprocessed, y_train)

            # print(grid.best_params_)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'best_params', grid.best_params_)

            ml_flow_client.log_metric(ml_flow_run.info.run_id, 'train_rmse', f'{-grid.best_score_}')
            # print(f'rmse: {-grid.best_score_}')

            best_model = grid.best_estimator_

            print(f'X_validate/y SHAPE: {X_validate.shape}/{y_validate.shape}')
            validate_rmse = self.evaluate(best_model, X_val_preprocessed, y_validate)
            ml_flow_client.log_metric(ml_flow_run.info.run_id, 'validate_rmse', f'{validate_rmse}')
            print(f"validate_rmse: {validate_rmse}")

            # df_test = get_test_data()
            y_pred = best_model.predict(X_test_preprocessed)
            print(f'y_pred: {y_pred}')
            print(f'X_test: {X_test.columns}')
            # print(y_pred_list)
            self.save_submission(y_pred, X_test["key"], estimator_name, process_start_time)
            self.save_model(best_model, estimator_name, process_start_time)

            loop_end_time = time.time()
            loop_elapsed_time = round((loop_end_time - loop_start_time), 1)
            ml_flow_client.log_metric(ml_flow_run.info.run_id, 'elapsed_time', f'{loop_elapsed_time}')

    def evaluate(self, model, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = model.predict(X_test)
        rmse = utils.compute_rmse(y_pred, y_test)
        return round(rmse, 2)

    def save_model(self, model, estimator_name, process_start_time):
        file_name = f'gcp_model_{estimator_name}_{process_start_time}.joblib'
        local_file_path = gcp_params.LOCAL_STORAGE_LOCATION + file_name
        print(f'model local_file_path: {local_file_path}')

        joblib.dump(model, local_file_path)

        client = storage.Client()
        bucket = client.bucket(gcp_params.BUCKET_NAME)
        blob = bucket.blob(gcp_params.GCM_STORAGE_LOCATION + file_name)

        blob.upload_from_filename(gcp_params.LOCAL_STORAGE_LOCATION + file_name)
        print(f"uploaded {gcp_params.LOCAL_STORAGE_LOCATION}{file_name} => {gcp_params.GCM_STORAGE_LOCATION}/{file_name}")

        # self.upload_model_to_gcp(file_name)

    # def upload_model_to_gcp(self, file_name):

    #     client = storage.Client()
    #     bucket = client.bucket(gcp_params.BUCKET_NAME)
    #     blob = bucket.blob(gcp_params.MODEL_STORAGE_LOCATION + file_name)

    #     blob.upload_from_filename(gcp_params.SUBMISSION_STORAGE_LOCATION + file_name)
    #     print(f"uploaded {gcp_params.SUBMISSION_STORAGE_LOCATION}{file_name} => {gcp_params.SUBMISSION_STORAGE_LOCATION}/{file_name}")

    def save_submission(self, y_pred, y_keys, estimator_name, process_start_time):

        client = storage.Client()
        # bucket = client.bucket(gcp_params.BUCKET_NAME)

        y_pred = pd.concat([y_keys, pd.Series(y_pred)],axis=1)
        y_pred.columns = ['key', 'fare_amount']
        file_name = f'submission_{estimator_name}_{process_start_time}.csv'
        local_file_path = gcp_params.LOCAL_STORAGE_LOCATION + file_name
        print(f'submission local_file_path: {local_file_path}')

        #save locally
        pd.DataFrame(y_pred).to_csv(local_file_path, index=False)

        #save go GCP, no need for local file save even though one occurs above
        f = io.StringIO()
        y_pred.to_csv(f)
        f.seek(0)
        client.get_bucket(gcp_params.BUCKET_NAME).blob(gcp_params.GCM_STORAGE_LOCATION + file_name).upload_from_file(f, content_type='text/csv')


        # client = storage.Client()
        # bucket = client.bucket(gcp_params.BUCKET_NAME)
        # blob = bucket.blob(gcp_params.MODEL_STORAGE_LOCATION + file_name)

        # blob.upload_from_filename(gcp_params.SUBMISSION_STORAGE_LOCATION + file_name)
        # print(f"uploaded {gcp_params.SUBMISSION_STORAGE_LOCATION}{file_name} => {gcp_params.SUBMISSION_STORAGE_LOCATION}/{file_name}")


    # MLFlow methods
    # @memoized_property
    # def mlflow_client(self):
    #     print('mlflow_client')
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     print('mlflow_experiment_id')
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         id = self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id
    #         print(f'mlflow_experiment_id: {id}')
    #         return id

    # @memoized_property
    # def mlflow_run(self):
    #     print('mlflow_run')
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     print(f'mlflow_log_param: {key}:{value}')
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     print(f'mlflow_log_metric: {key}:{value}')
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":

    # starttime = int(round(time.time() * 1000))


    print(f'Number of arguments: {len(sys.argv)}')
    print(f'Arguments: {sys.argv}')

    params = {
        'estimators': {
            'knn':{
                'hyperparams':{
                    'n_neighbors':[10,20,50],
                    'n_jobs':[-1],
                },
            },
            'linear':{
                'hyperparams':{
                    'n_jobs':[-1],
                }
            },
            'sgd':{
                'hyperparams':{
                    'learning_rate': ['invscaling'],
                }
            }
        },
        # 'nrows':1_000,
        'nrows':200_000,
        # 'starttime':starttime,
        'experiment_name':EXPERIMENT_NAME
    }

    trainer = Trainer(params)
    trainer.run()
        # trainer.set_experiment_name(EXPERIMENT_NAME)
        # trainer.run()



    # Get and clean data
    # N = 100_000
    # df = get_raw_data(nrows=N)

    # df = clean_data(df)
    # y = df["fare_amount"]
    # X = df.drop("fare_amount", axis=1)
    # print(f'X/y SHAPE: {X.shape}/{y.shape}')


    # for estimator, params in estimators.items():
    #     print(f'--{k}::{v}')

    # #BEGIN estimators LOOP
    # print(f'Xtrain/y SHAPE: {X_train.shape}/{y_train.shape}')
    # trainer = Trainer(estimator, params, X=X_train, y=y_train)
    # trainer.set_experiment_name(EXPERIMENT_NAME)
    # trainer.run()

    # print(f'Xtest/y SHAPE: {X_test.shape}/{y_test.shape}')
    # rmse = trainer.evaluate(X_test, y_test)
    # print(f"rmse: {rmse}")

    # df_test = get_test_data()
    # y_pred = trainer.predict(df_test)

    # # print(y_pred_list)
    # trainer.save_submission(y_pred)
    # trainer.save_model(trainer.pipeline)
    #END estimators  LOOP


    # for k,v in estimators.items():
    #     print(f'--{k}::{v}')





    # put date time stamp here to do long running trainings and cross _Validation_records
    # while saving the models (with same timestamp) log metrics of runs/CVs and save
    # the models and metrics that created them

    # build cloud based grid search over different models with
    # different _Parameters

    #standardize/normalize input data

    #try different estimators/models
    # -- Try Neural Network
'''
Xload train/val data
Xfit train/val data
Xtransform train/val data
??split train/val datda

xload test data
xtransfrom test data


LOOP estimators
X    grid search train/val data

X    get best model

    store models
    evaluate test data on best model from GridSearch results

Update/c

from best/best model, submit prediction
'''
