# -*- coding: utf-8 -*-


"""# LOADING THE DATASET
In this notebook we will use the ratings.csv file to perform the benchmarking operations on several metrics like : RMSE, MAE and FIT_TIME. We will use pandas dataframe to read the file.

Let us now import the surprise package and convert to surprise readable format using the surprise.dataset function.
"""

import surprise
from surprise import Dataset
from surprise.reader import Reader
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise import accuracy
import numpy
import time
from pathlib import Path

def built_in() -> (surprise.dataset.DatasetAutoFolds):
  data = Dataset.load_builtin(name='ml-100k',prompt = True)
  return data

def user_defined_file(file: Path) -> (surprise.dataset.DatasetAutoFolds) :
  reader = Reader(line_format='user item rating timestamp',sep= ',',skip_lines=1)
  data = Dataset.load_from_file(file,reader)
  return data

def get_data(load_from_surprise: bool = True, ratings_filepath : Path = None) -> (surprise.dataset.DatasetAutoFolds):
  if load_from_surprise: 
    data = built_in()
  else:
    data = user_defined_file(ratings_filepath)
  return data

"""
# MODEL PIPELINE
We will be using algorithms, the KNNBasic, Baseline, SVD, SVDpp and NMF .
We will be using the default params or pass in the required arguments to the model.
"""
''' Let us now define the different algorithms required.'''
def knn_uc() -> KNNBasic:
  return KNNBasic(k=20,min_k=4,sim_options={'name':'cosine','user_based':True})

def knn_ic() -> KNNBasic:
  return KNNBasic(k=20,min_k=4,sim_options={'name':'cosine','user_based':False})

def knn_up() -> KNNBasic:
  return KNNBasic(k=20,min_k=4,sim_options={'name':'pearson','user_based':True})

def knn_ip() -> KNNBasic:
  return KNNBasic(k=20,min_k=4,sim_options={'name':'pearson','user_based':False})

def baseline_algo() -> BaselineOnly:
  user_input = input('Do you want to continue with the default parameters? Y/N')
  if user_input.lower() == 'y':
    return BaselineOnly()
  else:
    bsl_options = {}
    method = str(input('Which method do you want to proceed with? [SGD/ALS] '))
    if method.lower() == 'als':
      bsl_options['method'] = method
      n_epochs = int(input('Enter number of epochs'))
      reg_u = int(input(' Enter value of reg_u'))
      reg_i = int(input('Enter value of reg_i'))
      bsl_options['n_epochs'] = n_epochs
      bsl_options['reg_u'] = reg_u
      bsl_options['reg_i'] = reg_i
      return BaselineOnly(bsl_options=bsl_options)

    if method.lower() == 'sgd':
      bsl_options['method'] = method
      n_epochs = int(input('Enter number of epochs'))
      reg = int(input(' Enter value of reg'))
      learning_rate = float(input('Enter learning_rate'))
      bsl_options['n_epochs'] = n_epochs
      bsl_options['reg'] = reg
      bsl_options['learning_rate'] = learning_rate  
      return BaselineOnly(bsl_options=bsl_options)
    
    else:
      print('Method not found')
      return None

def svd_algorithm() -> SVD:
  user_input = input('Do you want to continue with the default parameters? Y/N')
  if user_input.lower() == 'y':
    return SVD()
  else:
    n_factors = int(input('Enter total number of factors: '))
    n_epochs = int(input('Enter number of epochs: '))
    lr_all = float(input('Enter the learning rate for all the paramaters: '))
    return SVD(n_factors,n_epochs,lr_all)

def svdpp_algorithm() -> SVDpp:
  user_input = input('Do you want to continue with the default parameters? Y/N')
  if user_input.lower() == 'y':
    return SVDpp()
  else:
    n_factors = int(input('Enter total number of factors: '))
    n_epochs = int(input('Enter number of epochs: '))
    lr_all = float(input('Enter the learning rate for all the paramaters: '))
    return SVDpp(n_factors,n_epochs,lr_all)

def nmf_algorithm() -> NMF:
  user_input = input('Do you want to continue with the default parameters? Y/N')
  if user_input.lower() == 'y':
    return NMF()
  else:
    n_factors = int(input('Enter total number of factors: '))
    n_epochs = int(input('Enter number of epochs: '))
    return NMF(n_factors,n_epochs)

''' AFTER DEFINING THE DIFFERENT ALGORITHMS, LET US NOW CREATE A FUNCTION THROUGH WHICH WE CAN CALL THE USER REQUESTED ALGORITHM'''
def train_model_pipeline(model: str) -> (KNNBasic or BaselineOnly or SVD or SVDpp or NMF):
  if model.lower() == 'knn_uc': 
    return knn_uc()
  if model.lower() == 'knn_ic':
    return knn_ic()
  if model.lower() == 'knn_up':
    return knn_up()
  if model.lower() == 'knn_ip':
    return knn_ip()
  if model.lower() == 'baseline':
    return baseline_algo()
  if model.lower() == 'svd':
    return svd_algorithm()
  if model.lower() == 'svdpp':
    return svdpp_algorithm()
  if model.lower() == 'nmf':
    return nmf_algorithm()

''' LET US NOW DEFINE A FUNCTION TO TRAIN AND PREDICT'''
def model_predict(model:str,train:surprise.trainset.Trainset,test:list) -> (list,float):
  start = time.time()
  model.fit(train)
  stop = time.time()
  train_time = stop - start
  predictions = model.test(test)
  return predictions,train_time

''' DEFINING A FUNCTION TO EVALUATE THE MODEL PERFORMANCE'''
def evaluate(pred:list) -> (numpy.float64, numpy.float64):
  return accuracy.rmse(pred),accuracy.mae(pred)

''' DEFINE A FUNCTION CALLED benchamrking_pipeline TO HANDLE ALL THE OPERATIONS TO PERFORM BENCHMARKING'''
def benchmarking_pipeline(model:[KNNBasic or BaselineOnly or SVD or SVDpp or NMF],traindata:surprise.trainset.Trainset,testdata:list) -> (numpy.float64, numpy.float64, float):
  model = train_model_pipeline(model)
  predictions, fit_time = model_predict(model,traindata,testdata)
  rmse,mae = evaluate(predictions)
  return rmse,mae,fit_time
