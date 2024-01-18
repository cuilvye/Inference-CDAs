#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Evaluate different inference models using the complete real testing data.

"""


import pandas as pd
import os, argparse

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from keras.models import model_from_json
from utils import Determine_features_RNN
from models import Performance_on_ValidData_SpeedUp_v1_Ylabel2
from priors import load_or_compute_priorSigma
from priors import  load_or_compute_loss2_dependencies
from priors import prepare_for_loss3
from baselines import Performance_on_ValidData_Baseline_DL_KL_Ylabel
from baselines import Performance_on_ValidData_Baseline_SL_KL_Ylabel
from baselines import Performance_on_ValidData_Baseline_DL_Ylabel
from baselines import Performance_on_ValidData_Baseline_SL_Ylabel
from baselines import Performance_on_ValidData_Baseline_1_Ylabel
from baselines import Performance_on_ValidData_Baseline_2_v0_Ylabel

import warnings
warnings.filterwarnings("ignore")


def load_trained_behavior_model(path_model):
    model_name = path_model + 'model.json'
    model_weights = path_model + 'model.h5'
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    return model

def load_trained_cost_model(path_model):
    model_name = path_model + 'model_cost.json'
    model_weights = path_model + 'model_cost.h5'
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    return model

print('*************************** Running Evaluation Exp on Real Test CDA Dataset ***************************')
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--file_path',  default ='./Data/', help = 'the root path of dataset', type = str)
args_parser.add_argument('--file_name', default ='Filtered_data_CDA_trans_IR', help = 'the dataset name', type = str)
args_parser.add_argument('--classes', default = 7, help = 'the class number of action', type = int)
args_parser.add_argument('--epochs', default = 500, help = 'the training epochs', type = int)
args_parser.add_argument('--batch', default = 64, help = 'the size of training batch ', type = int)
args_parser.add_argument('--lr', default = 0.001, help = 'the learning rate of optimizer ', type = float)
args_parser.add_argument('--disPars', default = {'price_min': 1, 'price_max': 300, 'gap': 3}, help = 'the distribution pars of dataset ', type = dict)
args_parser.add_argument('--split_idx', default = 1, help = 'different data split:train/valid/test', type = int)
args_parser.add_argument('--model_path', default = './TrainedModels/', help = 'save path', type = str)
args_parser.add_argument('--costType', default = 'Uniform', help = 'the prior distribution of Cost ', type = str)
args_parser.add_argument('--sigma', default = 0.82623, help = 'the parameter for computing prior distribution', type = float)
args = args_parser.parse_args()

# pars
split_idx = args.split_idx
y_classes = args.classes
epochs = args.epochs
bt_size = args.batch
lr = args.lr
dis_pars = args.disPars
model_path = args.model_path
cost_disType = args.costType
sigma = args.sigma  # fitting with a two-parameter distribution

file_name = args.file_name  # .csv
file_path = args.file_path
### pay attention to that CDA data format is different from Bargaining data,
### it is more similar to the format after converting_into_training_data..
### so there is no need to convert it anymore.
train_name = 'Train' + str(split_idx) + '_' + file_name
valid_name = 'Valid' + str(split_idx) + '_' + file_name
test_name = 'Test' + str(split_idx) + '_' + file_name
df_train_upper = pd.read_csv(file_path + train_name + '_UB.csv', header=0)
df_valid_upper = pd.read_csv(file_path + valid_name + '_UB.csv', header=0)
df_test_upper = pd.read_csv(file_path + test_name + '_UB.csv', header=0)

df_train_new, seq_dim_train = Determine_features_RNN(df_train_upper)
df_valid_new, seq_dim_valid = Determine_features_RNN(df_valid_upper)
df_test_new, seq_dim_test = Determine_features_RNN(df_test_upper)
timeStep = 3

save_root = './Eval/' + file_name + '/'
if not os.path.exists(save_root):
    os.makedirs(save_root)
resultsAcc_table = {'data-type': [],
                    'Ours-v1-newY': [],
                    'DL-KL': [],
                    'SL-KL': [],
                    'DL': [],
                    'SL': [],
                    'CrossEntropy': [],
                    'BLUE': []}
################################################# priors used by some algorithm, e.g., ours ###########################################################
root_path  = prior_path = model_path + '/'
save_priors_and_costSpace =  prior_path +'Data_Priors_CostSpace/'
if not os.path.exists(save_priors_and_costSpace):
    os.makedirs(save_priors_and_costSpace)
prior_name_train = save_priors_and_costSpace + train_name + '_PriorDis_OnSigma.json'
prior_name_valid = save_priors_and_costSpace + valid_name + '_PriorDis_OnSigma.json'
prior_name_test = save_priors_and_costSpace + test_name + '_PriorDis_OnSigma.json'

priorDis_dict_train = load_or_compute_priorSigma(prior_name_train, df_train_new, sigma, dis_pars)
priorDis_dict_valid = load_or_compute_priorSigma(prior_name_valid, df_valid_new, sigma, dis_pars)
priorDis_dict_test = load_or_compute_priorSigma(prior_name_test, df_test_new, sigma, dis_pars)

################################ Prepare the cost space for the computation of loss-2 and loss-3, to save time ################################
####### loss-2 first ####

file_dict_loss2_trainData = save_priors_and_costSpace + train_name + '_loss2_costSpace.json'
file_dict_loss2_trainData_priorDisArr = save_priors_and_costSpace + train_name + '_loss2_priorDisArr.json'
dict_loss2_trainData, dict_loss2_trainData_priorDisArr = load_or_compute_loss2_dependencies(file_dict_loss2_trainData,
                                                                                            file_dict_loss2_trainData_priorDisArr,
                                                                                            df_train_new, dis_pars,
                                                                                            prior_name_train, sigma)
file_dict_loss2_validData = save_priors_and_costSpace + valid_name + '_loss2_costSpace.json'
file_dict_loss2_validData_priorDisArr = save_priors_and_costSpace + valid_name + '_loss2_priorDisArr.json'
dict_loss2_validData, dict_loss2_validData_priorDisArr = load_or_compute_loss2_dependencies(file_dict_loss2_validData,
                                                                                            file_dict_loss2_validData_priorDisArr,
                                                                                            df_valid_new, dis_pars,
                                                                                            prior_name_valid, sigma)

file_dict_loss2_testData = save_priors_and_costSpace + test_name + '_loss2_costSpace.json'
file_dict_loss2_testData_priorDisArr = save_priors_and_costSpace + test_name + '_loss2_priorDisArr.json'
dict_loss2_testData, dict_loss2_testData_priorDisArr = load_or_compute_loss2_dependencies(file_dict_loss2_testData,
                                                                                            file_dict_loss2_testData_priorDisArr,
                                                                                            df_test_new, dis_pars,
                                                                                            prior_name_test, sigma)

####### loss-3 then ####
Path_dict_loss3_trainData_all = save_priors_and_costSpace + train_name + '_loss3_costSpace_all/'
Path_dict_loss3_trainData_cons = save_priors_and_costSpace + train_name + '_loss3_costSpace_cons/'
prepare_for_loss3(Path_dict_loss3_trainData_all, Path_dict_loss3_trainData_cons, df_train_new, dis_pars)

Path_dict_loss3_validData_all = save_priors_and_costSpace + valid_name + '_loss3_costSpace_all/'
Path_dict_loss3_validData_cons = save_priors_and_costSpace + valid_name + '_loss3_costSpace_cons/'
prepare_for_loss3(Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons, df_valid_new, dis_pars)

Path_dict_loss3_testData_all = save_priors_and_costSpace + test_name + '_loss3_costSpace_all/'
Path_dict_loss3_testData_cons = save_priors_and_costSpace + test_name + '_loss3_costSpace_cons/'
prepare_for_loss3(Path_dict_loss3_testData_all, Path_dict_loss3_testData_cons, df_test_new, dis_pars)


################# the performance with OURS model #################
####### new-improvements #######
print('==================================== Ours v1-newY Inference Performance =================================================')
weights = (0.2, 0.6, 0.2)
classes = 7
path_model = root_path + 'RNN_262_Split_' + str(split_idx) + '_KATE_Classes_' + str(classes) + '_models/'
model_ours = load_trained_behavior_model(path_model)
# train_costMse_mean, valid_costMse_mean, test_costMse_mean = 836.5731918558997, 1009.2909763265695, 862.138574350155
train_loss_mean, train_acc_mean, train_costMse_mean =  Performance_on_ValidData_SpeedUp_v1_Ylabel2(df_train_new, classes, bt_size, cost_disType, dis_pars, model_ours, timeStep, Path_dict_loss3_trainData_all, Path_dict_loss3_trainData_cons,
                                     dict_loss2_trainData, dict_loss2_trainData_priorDisArr, weights)
valid_loss_mean, valid_acc_mean, valid_costMse_mean =  Performance_on_ValidData_SpeedUp_v1_Ylabel2(df_valid_new, classes, bt_size, cost_disType, dis_pars, model_ours, timeStep, Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons,
                                     dict_loss2_validData, dict_loss2_validData_priorDisArr, weights)
test_loss_mean, test_acc_mean, test_costMse_mean =  Performance_on_ValidData_SpeedUp_v1_Ylabel2(df_test_new, classes, bt_size, cost_disType, dis_pars, model_ours, timeStep, Path_dict_loss3_testData_all, Path_dict_loss3_testData_cons,
                                     dict_loss2_testData, dict_loss2_testData_priorDisArr, weights)
resultsAcc_table['Ours-v1-newY'].append(round(train_costMse_mean, 4))
resultsAcc_table['Ours-v1-newY'].append(round(valid_costMse_mean, 4))
resultsAcc_table['Ours-v1-newY'].append(round(test_costMse_mean, 4))
print('train_cost_mse: {}, valid_cost_mse: {}, test_cost_mse: {}; '.format(train_costMse_mean,
                                                                            valid_costMse_mean,
                                                                            test_costMse_mean), end ='\n')
#
################# the performance with DL-KL model #################
print('==================================== DL-KL Inference Performance =================================================')
path_model_dl_kl = root_path + 'RNN_Split_' + str(split_idx) + '_DL_KL_Classes_' + str(classes) + '_models/'
model_y = load_trained_behavior_model(path_model_dl_kl)
model_cost = load_trained_cost_model(path_model_dl_kl)
train_loss_mean, train_loss_cost_mean, train_acc_mean, train_costMse_mean = Performance_on_ValidData_Baseline_DL_KL_Ylabel(df_train_new, priorDis_dict_train, classes, bt_size, model_y, model_cost, dis_pars, timeStep)
valid_loss_mean, valid_loss_cost_mean, valid_acc_mean, valid_costMse_mean = Performance_on_ValidData_Baseline_DL_KL_Ylabel(df_valid_new, priorDis_dict_valid, classes, bt_size, model_y, model_cost, dis_pars, timeStep)
test_loss_mean, test_loss_cost_mean, test_acc_mean, test_costMse_mean = Performance_on_ValidData_Baseline_DL_KL_Ylabel(df_test_new, priorDis_dict_test, classes, bt_size, model_y, model_cost, dis_pars, timeStep)
resultsAcc_table['DL-KL'].append(round(train_costMse_mean, 4))
resultsAcc_table['DL-KL'].append(round(valid_costMse_mean, 4))
resultsAcc_table['DL-KL'].append(round(test_costMse_mean, 4))
print('train_cost_mse: {}, valid_cost_mse: {}, test_cost_mse: {}; '.format(train_costMse_mean,
                                                                            valid_costMse_mean,
                                                                            test_costMse_mean), end ='\n')


################# the performance with SL-KL model #################
print('==================================== SL-KL Inference Performance =================================================')
path_model_sl_kl = root_path + 'RNN_Split_' + str(split_idx) + '_SL_KL_Classes_' + str(classes) + '_models/'
model_cost = load_trained_cost_model(path_model_sl_kl)
train_loss_mean,  train_costMse_mean = Performance_on_ValidData_Baseline_SL_KL_Ylabel(df_train_new, priorDis_dict_train, classes, bt_size, dis_pars, model_cost, timeStep)
valid_loss_mean,  valid_costMse_mean = Performance_on_ValidData_Baseline_SL_KL_Ylabel(df_valid_new, priorDis_dict_valid, classes, bt_size, dis_pars, model_cost, timeStep)
test_loss_mean,  test_costMse_mean = Performance_on_ValidData_Baseline_SL_KL_Ylabel(df_test_new, priorDis_dict_test, classes, bt_size, dis_pars, model_cost, timeStep)
resultsAcc_table['SL-KL'].append(round(train_costMse_mean, 4))
resultsAcc_table['SL-KL'].append(round(valid_costMse_mean, 4))
resultsAcc_table['SL-KL'].append(round(test_costMse_mean, 4))
print('train_cost_mse: {}, valid_cost_mse: {}, test_cost_mse: {}; '.format(train_costMse_mean,
                                                                           valid_costMse_mean,
                                                                           test_costMse_mean), end ='\n')

################# the performance with DL model #################
print('==================================== DL Inference Performance =================================================')
path_model_dl = root_path + 'RNN_Split_' + str(split_idx) + '_DL_Classes_' + str(classes) + '_models/'
model_y = load_trained_behavior_model(path_model_dl)
model_cost = load_trained_cost_model(path_model_dl)
train_loss_mean, train_loss_cost_mean, train_acc_mean, train_costMse_mean, train_costAcc_mean = Performance_on_ValidData_Baseline_DL_Ylabel(df_train_new, classes, bt_size, model_y, model_cost, dis_pars, timeStep)
valid_loss_mean, valid_loss_cost_mean, valid_acc_mean, valid_costMse_mean, valid_costAcc_mean = Performance_on_ValidData_Baseline_DL_Ylabel(df_valid_new, classes, bt_size, model_y, model_cost, dis_pars, timeStep)
test_loss_mean, test_loss_cost_mean, test_acc_mean, test_costMse_mean, test_costAcc_mean = Performance_on_ValidData_Baseline_DL_Ylabel(df_test_new, classes, bt_size, model_y, model_cost, dis_pars, timeStep)
resultsAcc_table['DL'].append(round(train_costMse_mean, 4))
resultsAcc_table['DL'].append(round(valid_costMse_mean, 4))
resultsAcc_table['DL'].append(round(test_costMse_mean, 4))
print('train_cost_mse: {}, valid_cost_mse: {}, test_cost_mse: {}; '.format(train_costMse_mean,
                                                                           valid_costMse_mean,
                                                                           test_costMse_mean), end ='\n')

print('==================================== SL Inference Performance =================================================')
path_model_sl = root_path + 'RNN_Split_' + str(split_idx) + '_SL_Classes_' + str(classes) + '_models/'
model_cost = load_trained_cost_model(path_model_sl)
train_loss_mean,  train_costMse_mean, train_costAcc_mean = Performance_on_ValidData_Baseline_SL_Ylabel(df_train_new, classes, bt_size, dis_pars, model_cost, timeStep)
valid_loss_mean, valid_costMse_mean, valid_costAcc_mean = Performance_on_ValidData_Baseline_SL_Ylabel(df_valid_new, classes, bt_size, dis_pars, model_cost, timeStep)
test_loss_mean,  test_costMse_mean, test_costAcc_mean = Performance_on_ValidData_Baseline_SL_Ylabel(df_test_new, classes, bt_size, dis_pars, model_cost, timeStep)
resultsAcc_table['SL'].append(round(train_costMse_mean, 4))
resultsAcc_table['SL'].append(round(valid_costMse_mean, 4))
resultsAcc_table['SL'].append(round(test_costMse_mean, 4))
print('train_cost_mse: {}, valid_cost_mse: {}, test_cost_mse: {}; '.format(train_costMse_mean,
                                                                           valid_costMse_mean,
                                                                           test_costMse_mean), end ='\n')


print('==================================== Cross Entropy Inference Performance =================================================')
path_model_loss1 = root_path + 'RNN_Split_' + str(split_idx) + '_CE_Classes_' + str(classes) + '_models/'
model = load_trained_behavior_model(path_model_loss1)
train_loss_mean, train_acc_mean, train_costMse_mean = Performance_on_ValidData_Baseline_1_Ylabel(df_train_new, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_trainData_all)
valid_loss_mean, valid_acc_mean, valid_costMse_mean = Performance_on_ValidData_Baseline_1_Ylabel(df_valid_new, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_validData_all)
test_loss_mean, test_acc_mean, test_costMse_mean = Performance_on_ValidData_Baseline_1_Ylabel(df_test_new, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_testData_all)
resultsAcc_table['CrossEntropy'].append(round(train_costMse_mean, 4))
resultsAcc_table['CrossEntropy'].append(round(valid_costMse_mean, 4))
resultsAcc_table['CrossEntropy'].append(round(test_costMse_mean, 4))
print('train_cost_mse: {}, valid_cost_mse: {}, test_cost_mse: {}; '.format(train_costMse_mean,
                                                                           valid_costMse_mean,
                                                                           test_costMse_mean), end ='\n')

print('==================================== BLUE Inference Performance =================================================')
alpha = 0.6
path_model_blue = root_path + 'RNN_Split_' + str(split_idx) + '_BLUE_Classes_' + str(classes) + '_models/'
model = load_trained_behavior_model(path_model_blue)
train_loss_mean, train_acc_mean, train_costMse_mean = Performance_on_ValidData_Baseline_2_v0_Ylabel(df_train_new, classes, bt_size, cost_disType, dis_pars,model, timeStep, alpha, Path_dict_loss3_trainData_all)
valid_loss_mean, valid_acc_mean, valid_costMse_mean = Performance_on_ValidData_Baseline_2_v0_Ylabel(df_valid_new, classes, bt_size, cost_disType, dis_pars,model, timeStep, alpha, Path_dict_loss3_validData_all)
test_loss_mean, test_acc_mean, test_costMse_mean = Performance_on_ValidData_Baseline_2_v0_Ylabel(df_test_new, classes, bt_size, cost_disType, dis_pars,model, timeStep, alpha, Path_dict_loss3_testData_all)
resultsAcc_table['BLUE'].append(round(train_costMse_mean, 4))
resultsAcc_table['BLUE'].append(round(valid_costMse_mean, 4))
resultsAcc_table['BLUE'].append(round(test_costMse_mean, 4))
print('train_cost_mse: {}, valid_cost_mse: {}, test_cost_mse: {}; '.format(train_costMse_mean,
                                                                           valid_costMse_mean,
                                                                           test_costMse_mean), end ='\n')


############# save the result file #############
resultsAcc_table['data-type'].append('train')
resultsAcc_table['data-type'].append('valid')
resultsAcc_table['data-type'].append('test')

pd.DataFrame(resultsAcc_table).to_csv(save_root +'Results_Final_Split_' + str(split_idx) + '_CostMses.csv', index = False)












