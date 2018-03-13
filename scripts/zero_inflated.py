# load classification and regression results
# do elemen-wise multiplication of these results
# both propabilities and hardcut
# get summaries and save pickle
# save the index of filtered observations
# ****************************************************************
# library import block
# ****************************************************************
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import logging
import time
import sys
from scipy.cluster.vq import kmeans
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.switch_backend('agg')

float_type = tf.float64
jitter_level = 1e-5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(scriptPath):
    tf.reset_default_graph()
    parentDir = '/'.join(os.path.dirname(os.path.realpath(scriptPath)).split('/')[:-1])  #
    subDir = "/" + scriptPath.split("/")[-2].split(".py")[0] + "/"
    sys.path.append(parentDir)

    clf_modelPath = parentDir + subDir + 'results_scgp.pickle'
    reg_modelPath = parentDir + subDir + 'results_svgp.pickle'
    logPath   = parentDir + subDir + 'modelsumm_zi.log'

    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(logPath))

    data = pickle.load(open(parentDir + subDir +"data.pickle","rb"))
    Xtrain = data['Xtrain']
    Ytrain = data['Ytrain']
    Xtest = data['Xtest']
    Ytest = data['Ytest']

    # load results from the clasiifier model
    clf_results = pickle.load(open(clf_modelPath,"rb"))
    reg_results = pickle.load(open(reg_modelPath,"rb"))

    # combined results

    # ****************************************************************
    # model predictions
    # ****************************************************************
    train_clf_prob = clf_results['pred_train']['pfmean']
    test_clf_prob = clf_results['pred_test']['pfmean']
    train_clf_indc = (train_clf_prob > 0.5) * 1.0
    test_clf_indc = (test_clf_prob > 0.5) * 1.0

    pred_train_zi_prob = train_clf_prob * reg_results['pred_train']['fmean']
    pred_test_zi_prob = test_clf_prob * reg_results['pred_test']['fmean']
    pred_train_zi_indc = train_clf_indc * reg_results['pred_train']['fmean']
    pred_test_zi_indc = test_clf_indc * reg_results['pred_test']['fmean']

    def rmse(predict,actual):
        predict = np.maximum(predict,0)
        return np.sqrt(np.mean((actual-predict)**2))

    def mad(predict,actual):
        predict = np.maximum(predict,0)
        return np.mean(np.abs(actual-predict))

    train_zi_prob_reg_rmse = rmse(pred_train_zi_prob,Ytrain)
    logger.info("rmse on train set for zi prob : "+str(train_zi_prob_reg_rmse))
    train_zi_prob_reg_mae = mad(pred_train_zi_prob,Ytrain)
    logger.info("mae on train set for zi prob : "+str(train_zi_prob_reg_mae))

    test_zi_prob_reg_rmse = rmse(pred_test_zi_prob,Ytest)
    logger.info("rmse on test set for zi prob  : "+str(test_zi_prob_reg_rmse))
    test_zi_prob_reg_mae = mad(pred_test_zi_prob,Ytest)
    logger.info("mae on test set for zi prob  : "+str(test_zi_prob_reg_mae))

    train_zi_indc_reg_rmse = rmse(pred_train_zi_indc,Ytrain)
    logger.info("rmse on train set for zi indc : "+str(train_zi_indc_reg_rmse))
    train_zi_indc_reg_mae = mad(pred_train_zi_indc,Ytrain)
    logger.info("mae on train set for zi indc : "+str(train_zi_indc_reg_mae))

    test_zi_indc_reg_rmse = rmse(pred_test_zi_indc,Ytest)
    logger.info("rmse on test set for zi indc  : "+str(test_zi_indc_reg_rmse))
    test_zi_indc_reg_mae = mad(pred_test_zi_indc,Ytest)
    logger.info("mae on test set for zi indc  : "+str(test_zi_indc_reg_mae))

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # ****************************************************************
    # return values
    # ****************************************************************
    results = {
               'pred_train_zi_prob':pred_train_zi_prob,
               'pred_test_zi_prob':pred_test_zi_prob,
               'pred_train_zi_indc':pred_train_zi_indc,
               'pred_test_zi_indc':pred_test_zi_indc,
               'train_zi_prob_reg_rmse':train_zi_prob_reg_rmse,
               'train_zi_prob_reg_mae':train_zi_prob_reg_mae,
               'test_zi_prob_reg_rmse':test_zi_prob_reg_rmse,
               'test_zi_prob_reg_mae':test_zi_prob_reg_mae,
               'train_zi_indc_reg_rmse':train_zi_indc_reg_rmse,
               'train_zi_indc_reg_mae':train_zi_indc_reg_mae,
               'test_zi_indc_reg_rmse':test_zi_indc_reg_rmse,
               'test_zi_indc_reg_mae':test_zi_indc_reg_mae
               }
    pickle.dump(results,open(parentDir+ subDir +"results_zi.pickle","wb"))


if __name__ == "__main__":
    scriptPath = sys.argv[0]
    main(scriptPath)
