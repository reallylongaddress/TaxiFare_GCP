# from msilib.schema import _Validation_records
# from sqlite3.dbapi2 import _Parameters
# from this import d
import pandas as pd
import joblib
from termcolor import colored
import mlflow
import time

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
EXPERIMENT_NAME = "[REMOTE] [reallylongaddress] TaxiFareModel_GCP + 0.0.10"
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

        starttime = int(round(time.time() * 1000))

        # print('Trainer.run')
        self.data_pipeline = TaxiFare_GCP.data.get_preprocessing_pipeline()
        # print(f'type: {type(self.data_pipeline)}')
        df_train_val = TaxiFare_GCP.data.train_val_get_raw_data(self.params.get('nrows'))

        X_train_val = df_train_val.drop(columns=['fare_amount'])
        y_train_val = df_train_val['fare_amount']

        X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, test_size=0.20)

        self.data_pipeline.fit(X_train)

        X_train_preprocessed = self.data_pipeline.transform(X_train)
        X_val_preprocessed = self.data_pipeline.transform(X_validate)

        X_test_preprocessed = self.data_pipeline.transform(TaxiFare_GCP.data.test_get_raw_data())

        for estimator, hyperparams in self.params.get('estimators').items():
            print(f'key: {estimator}')

            ml_flow_client = MlflowClient()
            ml_flow_run = ml_flow_client.create_run(self.mlflow_experiment_id)

            print(f'mlflow_log_param: start_time:{starttime}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'starttime', f'{starttime}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'model', estimator)
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'train_size', f'{X_train_preprocessed.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'validate_size', f'{X_val_preprocessed.shape[0]}')
            ml_flow_client.log_param(ml_flow_run.info.run_id, 'test_size', f'{X_test_preprocessed.shape[0]}')


            #dbd todo - this is ugly, reduce to 1 line
            for param_key, param_value in hyperparams.items():
                ml_flow_client.log_param(ml_flow_run.info.run_id, param_key, param_value)
            print(f'self.mlflow_experiment_id: {self.mlflow_experiment_id}')

            grid = None
            model = None
            if estimator == 'knn':
                model = KNeighborsRegressor()

            elif estimator == 'sgd':
                model = SGDRegressor()

            elif estimator == 'linear':
                model = LinearRegression()
            else:
                raise Exception("Unknown model type")

            print(f'===={estimator}: {len(hyperparams)}')
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
            print(f"validate_rmse: {validate_rmse}")

            # df_test = get_test_data()
            y_pred = best_model.predict(X_test_preprocessed)
            print(f'y_pred: {y_pred}')
            # print(y_pred_list)
            trainer.save_submission(y_pred)
            trainer.save_model(trainer.pipeline)



    def evaluate(self, model, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = model.predict(X_test)
        rmse = utils.compute_rmse(y_pred, y_test)
        return round(rmse, 2)

    # def predict(self, X_test):
    #     return self.data_pipeline.predict(X_test)
dbd todo
    def save_model(self, reg):
        joblib.dump(reg, gcp_params.MODEL_NAME)
        print(f"saved {gcp_params.MODEL_NAME} locally")

        # # Implement here
        self.upload_model_to_gcp()
        print(f"uploaded {gcp_params.MODEL_NAME} to gcp cloud storage under \n => {gcp_params.STORAGE_LOCATION}/{gcp_params.MODEL_NAME}")
        pass
dbd todo
    def upload_model_to_gcp(self):

        client = storage.Client()
        bucket = client.bucket(gcp_params.BUCKET_NAME)
        blob = bucket.blob(f'{gcp_params.STORAGE_LOCATION}{gcp_params.MODEL_NAME}')

        blob.upload_from_filename(gcp_params.MODEL_NAME)
        pass

    def save_submission(self, y_pred, estimator='KNN'):

        # y_pred = pd.concat([df_test["key"],pd.Series(y_pred)],axis=1)
        # y_pred.columns = ['key', 'fare_amount']
        # pd.DataFrame(y_pred).to_csv(f'./submission_{estimator}.csv', index=False)
        pass

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

    params = {
        'estimators': {
            'knn':{
                'hyperparams':{
                    'n_neighbors':[5,10,20],
                    'n_jobs':[1,-1],
                },
            },
            'linear':{
                'hyperparams':{
                    'n_jobs':[-1],
                }
            },
            'sgd':{
                'hyperparams':{
                    'learning_rate': ['constant', 'optimal', 'invscaling'],
                }
            }
        },
        'nrows':10000,
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

    get best model

    evaluate test data on best model from GridSearch results



from best/best model, submit prediction
'''
