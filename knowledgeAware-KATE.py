#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The Knowledge-Aware Learning-Based Inference Method: Knowledge-Aware Cost Inference.

Created on Mon Mar 20 16:05 2023
@author: lvye

"""

import pandas as pd
import numpy as np
import os, sys, time
import json, math
import argparse

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error

from models import build_rnn_GRUModel
from models import Performance_on_ValidData_SpeedUp_v1_Ylabel2
from models import Callback_EarlyStopping
from models import accuracy_compute
from utils import extract_batchData_sameMarketSeller
from utils import Convert_to_training_DataFormat_Class7
from utils import Compute_upper_bound_unit_cost
from utils import Determine_features_RNN
from utils import Compute_unique_mkSel_Pair
from inferences import Sample_Costs_ForLoss1
from inferences import insert_Costs_into_batchX
from inferences import Insert_AllCosts_3D_Format
from inferences import Correct_logZero_Case
from inferences import Compute_Loss3_Component_BuiltIn
from inferences import CostVector_Inference_withMultiRows_withDict_BuiltIn
from inferences import compute_rows_costs_Yprobs
from priors import prepare_for_loss3
from priors import load_or_compute_loss2_dependencies

import warnings

warnings.filterwarnings("ignore")

def training_model_epochVS(df_Train, classes, dict_loss2_trainData, dict_loss2_trainData_priorDisArr,
                           Path_dict_loss3_trainData_all, Path_dict_loss3_trainData_cons, bt_size, model, optimizer,
                           cost_disType, dis_pars, epoch, path_res, timeStep, weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    train_loss, train_acc, train_costMse = [], [], []
    train_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_Train, bt_size)
    for batch in range(len(batch_dataFrame)):
        time_s = time.time()
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting.
        X_train, Y_train, X_train_Cost, X_train_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat_Class7(batch_idataFrame, classes)
        Y_train = to_categorical(Y_train, classes)
        X_train = X_train.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_train, Y_train,
                                                                                                df_CostVect_ID,cost_disType,
                                                                                                Path_dict_loss3_trainData_all, timeStep)
        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        batch_costVec_AvgError = mean_squared_error(X_train_Cost,CostsArray_Inferred)

        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_train_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        #### should use sampled version!!!!
        with tf.GradientTape() as tape:
            ############################################  Loss-1: cross-entropy computation ############################################
            X_BatchTensor = tf.convert_to_tensor(X_train, dtype=float)
            y_tensorBatch = tf.convert_to_tensor(Y_train)
            CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
            X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)

            cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
            logits = tf.clip_by_value(logits_org, 1e-10, 1)

            loss_1 = cce(y_tensorBatch, logits)  #
            acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))
            dim_EachStep = int(X_train.shape[1] / timeStep)

            #######################################  Loss-3 AND Loss-2 computation ######################################
            loss3_Set = []
            loss3_Print = []
            mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
            for mk, sel in mkSel_uniqueSet:
                df_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
                mkSel_idxSet = df_mkSel.index.tolist()

                mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
                # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
                with open(Path_dict_loss3_trainData_all + mkSel_name + '.json', 'r') as f:
                    costArray_InsertSet = json.load(f)

                with open(Path_dict_loss3_trainData_cons + mkSel_name + '.json', 'r') as fn:
                    costArray_InsertSet_Cons = json.load(fn)

                # costArray_InsertSet_Cons = dict_loss3_trainData_cons[str((mk, sel))]
                X_mk_sel = X_train[mkSel_idxSet, :]  #
                Y_mk_sel = Y_train[mkSel_idxSet, :]
                Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

                x_extends_TensorAll = Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep)
                Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx,model)
                x_extends_TensorCons = Insert_AllCosts_3D_Format(costArray_InsertSet_Cons, X_mk_sel, dim_EachStep)
                Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,model)

                loss3_Set = loss3_Set + [Grad_TensorAll - Grad_TensorCons]  # * df_mkSel.shape[0]

                ##### loss3 Print ######
                Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(
                    max([np.sum(Efun_c_Arr), 1.0e-10]))
                loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

                ########################################## loss2 Set and loss2 Print #################################
                df_mkSel_reIdx = df_mkSel.reset_index(drop=True)
                UnitsID_SetUnique = list(np.unique(df_mkSel_reIdx['unit'])) ### since not use list(set()), so the order is kept!
                mkSelUnit_idxSet_Set = [df_mkSel_reIdx.loc[df_mkSel_reIdx['unit'] == unit].index.tolist() for unit in UnitsID_SetUnique]
                ###### the inqury takes too long time?

                X_mkSelunit_set = [X_mk_sel[unitIdx_set, :] for unitIdx_set in mkSelUnit_idxSet_Set]
                # Y_mkSelunit_set = [Y_mk_sel[unitIdx_set, :] for unitIdx_set in mkSelUnit_idxSet_Set]
                Y_mk_sel_unit_trueIdx_set = [list(np.array(Y_mk_sel_trueIdx)[unitIdx_set]) for unitIdx_set in mkSelUnit_idxSet_Set]

                UnitCosts_InsertSet = [dict_loss2_trainData[str((mk, sel, unit))] for unit in UnitsID_SetUnique]
                assert len(UnitCosts_InsertSet) == len(UnitsID_SetUnique) ## the insert cost vector of each unit can be different!

                x_extends_Tensor_set = [Insert_AllCosts_3D_Format(cost_uInsert_Set, X_mk_sel_unit, dim_EachStep) for cost_uInsert_Set, X_mk_sel_unit in zip(UnitCosts_InsertSet, X_mkSelunit_set)]
                Y_result_Tensor_set = [compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx) for x_extends_Tensor,Y_mk_sel_unit_trueIdx in zip(x_extends_Tensor_set, Y_mk_sel_unit_trueIdx_set)]

                priorDis_mkSelUnit_realArr_set = [np.array(dict_loss2_trainData_priorDisArr[str((mk, sel, unit))]) for unit in UnitsID_SetUnique]
                def compute_loss2(Y_result_Tensor, priorDis_mkSelUnit_realArr):
                    Y_result_Tensor = tf.clip_by_value(Y_result_Tensor, 1e-10, 1)
                    Y_result_Tensor_log = tf.math.log(Y_result_Tensor)

                    grad_1 = - Y_result_Tensor_log * priorDis_mkSelUnit_realArr
                    Gfun_c_Arr = Y_result_Tensor.numpy()
                    comp_2nd = tf.reduce_sum(Y_result_Tensor_log * Gfun_c_Arr, axis = 0) / np.sum(Gfun_c_Arr, axis = 0)

                    grad_2 = priorDis_mkSelUnit_realArr * comp_2nd
                    grad = tf.reduce_sum(grad_1 + grad_2) # ERR reason!! -grad_1 + grad_2

                    entropy_prior = np.sum(priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                    entropy_prior_units = entropy_prior * Y_result_Tensor.get_shape()[1] # the latter is the row number for this unit
                    entropy_cross = np.sum(priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr, axis = 0))))
                    return (grad, entropy_prior_units - entropy_cross)

                result_loss2 = [compute_loss2(Y_result_Tensor, priorDis_mkSelUnit_realArr) for Y_result_Tensor, priorDis_mkSelUnit_realArr in zip(Y_result_Tensor_set, priorDis_mkSelUnit_realArr_set)]
                loss2_Set, loss2_Print = map(list, zip(*result_loss2))

            Grad_BatchSum = w1 * tf.reduce_mean(loss_1) + w2 * tf.reduce_mean(loss2_Set) + w3 * tf.reduce_mean(loss3_Set)  #
            Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)  #

        gradients = tape.gradient(Grad_BatchSum, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        # print( '*********************************************************** batch error *************************************************************************')
        time_e = time.time()
        if (batch + 1) % 50 == 0 or (batch + 1) == len(batch_dataFrame):
            print("[INFO] training batch {}/{}, total_loss: {:.5f}, class_acc: {:.5f}, cost_mse: {:.5f} ".format(batch + 1, len(batch_dataFrame), Print_loss_mean, acc_batch, batch_costVec_AvgError), end="\n")
            # print('Batch row number: {}'.format(X_train.shape[0]))
            print("This BATCH training took about {:.4} minutes".format((time_e - time_s)/60.0), end="\n")

        train_acc.append(acc_batch)
        train_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        train_loss.append(Print_loss_mean)

        train_epoch_log['batch'].append(batch + 1)
        train_epoch_log['acc'].append(acc_batch)
        train_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        train_epoch_log['total_loss'].append(Print_loss_mean)
        # pd.DataFrame(train_epoch_log).to_csv(path_res +'data_iter_'+ str(it) +'_g_'+str(g) +'_train_epoch_'+ str(epoch) +'.csv', index = False)

    train_loss_mean = np.mean(train_loss)
    train_acc_mean = np.mean(train_acc)
    train_costMse_mean = np.mean(train_costMse)

    return model, train_loss_mean, train_acc_mean, train_costMse_mean


def build_behavior_model(path_model, classes, input_dim, step, mask_val):
    if os.path.exists(path_model + 'model.json'):
        print("[INFO] ================ reloading the model with classes {} ==================".format(classes),
              end="\n")
        model_name = path_model + 'model.json'
        model_weights = path_model + 'model.h5'
        json_file = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_weights)
    else:
        print("[INFO] ================ creating the model with classes {} ==================".format(classes), end="\n")
        model = build_rnn_GRUModel(classes, input_dim, step, mask_val)

    return model

def learning_model_paras(df_train_upper, df_valid_upper, classes, sigma, epochs, batch_size, lr, cost_disType, dis_pars,
                         path_res, path_model, train_name, valid_name, save_root, weights):

    ###### determine the features which are used in RNN ######
    # need to verify that a1=\min{a1, a2, a3} and b1=\max{b1, b2, b3}
    df_train_new, seq_dim_train = Determine_features_RNN(df_train_upper)
    df_valid_new, seq_dim_valid = Determine_features_RNN(df_valid_upper)

    assert seq_dim_train == seq_dim_valid
    ############################################  building or reloading the model ##################################################
    timeStep = 3
    mask_val = 0
    input_dim = int(seq_dim_train / timeStep) + 1  # 8 + 1 (1: private cost), Refer to Determine_features_RNN.py function
    model = build_behavior_model(path_model, classes, input_dim, timeStep, mask_val)

    optimizer = Adam(learning_rate=lr, amsgrad=True)  # RMSprop
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    ################################ compute the prior distribution for each unit cost variable according to parameter \sigma ################################

    # be careful for the prior distribution of cost,i.e., range !!!!!
    save_priors_and_costSpace = save_root + 'Data_Priors_CostSpace/'
    if not os.path.exists(save_priors_and_costSpace):
        os.makedirs(save_priors_and_costSpace)
    prior_name_train = save_priors_and_costSpace + train_name + '_PriorDis_OnSigma.json'
    # used for computing loss-2, and this probs are saved in file_dict_loss2_trainData_priorDisArr !!!!!!
    prior_name_valid = save_priors_and_costSpace + valid_name + '_PriorDis_OnSigma.json'
    # priorDis_dict_train = load_or_compute_priorSigma(prior_name_train, df_train_new, sigma, dis_pars)
    # priorDis_dict_valid = load_or_compute_priorSigma(prior_name_valid, df_valid_new, sigma, dis_pars)

    ############ Shelve is very dependent on the platform: it is work on linux not for macOS   ############
    # priorDis_dict = Compute_Prior_distribution_unit_cost_v1(df_train_new, sigma, dis_pars)
    # if not os.path.exists(prior_name):
    # Compute_Prior_distribution_unit_cost_v2(df_train_new, sigma, dis_pars, prior_name)
    # with shelve.open(prior_name) as priorDis_dict:
    #    print(len(priorDis_dict))

    ################################ Prepare the cost space for the computation of loss-2 and loss-3, to save time ################################
    ####### loss-2 first ####

    file_dict_loss2_trainData = save_priors_and_costSpace + train_name + '_loss2_costSpace.json'
    file_dict_loss2_trainData_priorDisArr = save_priors_and_costSpace + train_name + '_loss2_priorDisArr.json'
    dict_loss2_trainData, dict_loss2_trainData_priorDisArr = load_or_compute_loss2_dependencies(file_dict_loss2_trainData,
                                                                                      file_dict_loss2_trainData_priorDisArr,
                                                                                      df_train_new, dis_pars, prior_name_train, sigma)
    file_dict_loss2_validData = save_priors_and_costSpace + valid_name + '_loss2_costSpace.json'
    file_dict_loss2_validData_priorDisArr = save_priors_and_costSpace + valid_name + '_loss2_priorDisArr.json'
    dict_loss2_validData, dict_loss2_validData_priorDisArr = load_or_compute_loss2_dependencies(file_dict_loss2_validData,
                                                                                      file_dict_loss2_validData_priorDisArr,
                                                                                      df_valid_new, dis_pars, prior_name_valid, sigma)

    ####### loss-3 then ####
    # file_dict_loss3_trainData_all = path_res + train_name + '_loss3_costSpace_all.json' # it is too large!!
    # file_dict_loss3_trainData_cons = path_res + train_name + '_loss3_costSpace_cons.json'
    Path_dict_loss3_trainData_all = save_priors_and_costSpace + train_name + '_loss3_costSpace_all/'
    Path_dict_loss3_trainData_cons = save_priors_and_costSpace + train_name + '_loss3_costSpace_cons/'
    prepare_for_loss3(Path_dict_loss3_trainData_all, Path_dict_loss3_trainData_cons, df_train_new, dis_pars)

    Path_dict_loss3_validData_all = save_priors_and_costSpace + valid_name + '_loss3_costSpace_all/'
    Path_dict_loss3_validData_cons = save_priors_and_costSpace + valid_name + '_loss3_costSpace_cons/'
    prepare_for_loss3(Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons, df_valid_new, dis_pars)

    ###############################################   start training model #################################
    start = 0
    if os.path.exists(path_res + 'data_train_epochs_' + str(epochs) + '.csv'):
        org_train_log = pd.read_csv(path_res + 'data_train_epochs_' + str(epochs) + '.csv', header=0)
        if org_train_log.shape[0] > 0:
            start = int(np.array(org_train_log)[-1, 0])
        train_log = org_train_log.to_dict(orient="list")
    else:
        train_log = {'epoch': [],
                     'train_lossTotal': [],
                     'train_acc': [],
                     'train_costMse': [],
                     'valid_lossTotal': [],
                     'valid_acc': [],
                     'valid_costMse': []}
    print(
        "[INFO] ================ starting training the inference model from epoch-{} ==================".format(start),
        end="\n")

    # priorDis_dict_valid = Compute_Prior_distribution_unit_cost_v1(df_valid_new, sigma, dis_pars)
    # save_dict_file(prior_name, priorDis_dict)

    for epoch in range(start, epochs):
        sys.stdout.flush()
        # loop over the data in batch size increments
        epochStart = time.time()
        model, loss_train, acc_train, mse_train = training_model_epochVS(df_train_new, classes,
                                                                         dict_loss2_trainData,
                                                                         dict_loss2_trainData_priorDisArr,
                                                                         Path_dict_loss3_trainData_all,
                                                                         Path_dict_loss3_trainData_cons,
                                                                         batch_size, model, optimizer, cost_disType,
                                                                         dis_pars,
                                                                         epoch, path_res, timeStep, weights)
        elapsed = (time.time() - epochStart) / 60.0
        print("[INFO] one epoch took {:.4} minutes".format(elapsed), end="\n")
        ## check if needing valid dataset to adjust the learning rate #
        loss_v, acc_v, mse_v = Performance_on_ValidData_SpeedUp_v1_Ylabel2(df_valid_new, classes, batch_size, cost_disType, dis_pars,
                                                                model, timeStep, Path_dict_loss3_validData_all,
                                                                Path_dict_loss3_validData_cons,
                                                                dict_loss2_validData, dict_loss2_validData_priorDisArr, weights)

        train_log['epoch'].append(epoch + 1)
        train_log['train_lossTotal'].append(loss_train)
        train_log['train_acc'].append(acc_train)
        train_log['train_costMse'].append(mse_train)

        train_log['valid_lossTotal'].append(loss_v)
        train_log['valid_acc'].append(acc_v)
        train_log['valid_costMse'].append(mse_v)
        print("[INFO] epoch {}/{}, train total_loss: {:.5f}, train accuracy: {:.5f}, train cost_mse: {:.5f}; ".format(
            epoch + 1, epochs, loss_train, acc_train, mse_train), end="\n")
        print("[INFO] epoch {}/{}, valid total_loss: {:.5f}, valid accuracy: {:.5f}, valid cost_mse: {:.5f};".format(
            epoch + 1, epochs, loss_v, acc_v, mse_v), end="\n")

        pd.DataFrame(train_log).to_csv(path_res + 'data_train_epochs_' + str(epochs) + '.csv', index=False)

        model_json = model.to_json()
        with open(path_model + 'model.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(path_model + 'model.h5')

        stopEarly = Callback_EarlyStopping(train_log['valid_lossTotal'], min_delta=0.01, patience=30)
        if stopEarly:
            print('Callback Earlystopping signal received at epoch: {}/{} '.format(epoch + 1, epochs))
            print('Terminating training ')
            break


print('*************************** Running KATE Exp on Real CDA Dataset ***************************')
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--file_path',  default ='./Data/', help = 'the root path of dataset', type = str)
args_parser.add_argument('--file_name', default = 'Filtered_data_CDA_trans_IR', help = 'the dataset name', type = str)
args_parser.add_argument('--classes', default = 7, help = 'the class number of action', type = int)
args_parser.add_argument('--sigma', default = 0.82623, help = 'the parameter for computing prior distribution', type = float)
args_parser.add_argument('--epochs', default = 500, help = 'the training epochs', type = int)
args_parser.add_argument('--batch', default = 64, help = 'the size of training batch ', type = int)
args_parser.add_argument('--lr', default = 0.001, help = 'the learning rate of optimizer ', type = float)
args_parser.add_argument('--costType', default = 'Uniform', help = 'the prior distribution of Cost ', type = str)
args_parser.add_argument('--disPars', default = {'price_min': 1, 'price_max': 300, 'gap': 3}, help = 'the distribution pars of dataset ', type = dict)
args_parser.add_argument('--split_idx', default = 1, help = 'different data split:train/valid/test', type = int)
args_parser.add_argument('--save_path', default = './RetrainingModels/', help = 'save path', type = str)
args_parser.add_argument('--orgWeights', default = 1, help = 'the weights of the three parts', type = float)
args = args_parser.parse_args()


split_idx = args.split_idx
classes = args.classes
epochs = args.epochs
batch_size = args.batch
lr = args.lr
cost_disType = args.costType
dis_pars = args.disPars
sigma = args.sigma  # fitting with a two-parameter distribution
orgWeights = args.orgWeights
save_path = args.save_path

if orgWeights == 1:
    weights = (0.2, 0.6, 0.2)
else:
    weights = (0.3, 0.4, 0.3)

file_name = args.file_name
file_path = args.file_path #'./data/'
train_name = 'Train' + str(split_idx) + '_' + file_name
valid_name = 'Valid' + str(split_idx) + '_' + file_name

if not os.path.exists(file_path + train_name + '_UB.csv'):
    df_train = pd.read_csv(file_path + train_name + '.csv', header=0)
    df_valid = pd.read_csv(file_path + valid_name + '.csv', header=0)
    ### fill nan with 0 for entire DataFrame ###
    df_train = df_train.fillna(0)
    df_valid = df_valid.fillna(0)
    df_train_upper = Compute_upper_bound_unit_cost(df_train)  # add a new col: upper_bound
    df_valid_upper = Compute_upper_bound_unit_cost(df_valid)
    df_train_upper.to_csv(file_path + train_name + '_UB.csv', index=False)
    df_valid_upper.to_csv(file_path + valid_name + '_UB.csv', index=False)
else:
    df_train_upper = pd.read_csv(file_path + train_name + '_UB.csv', header=0)
    df_valid_upper = pd.read_csv(file_path + valid_name + '_UB.csv', header=0)

save_root = save_path + file_name + '/'
if weights == (0.2, 0.6, 0.2):
    path_model = save_root + 'RNN_262_Split_' + str(split_idx) + '_KATE_Classes_' + str(classes) + '_models/'
    path_res = save_root + 'RNN_262_Split_' + str(split_idx) + '_KATE_Classes_' + str(classes) + '_results/'
elif weights == (0.3, 0.4, 0.3):
    path_model = save_root + 'RNN_343_Split_' + str(split_idx) + '_KATE_Classes_' + str(classes) + '_models/'
    path_res = save_root + 'RNN_343_Split_' + str(split_idx) + '_KATE_Classes_' + str(classes) + '_results/'
if not os.path.exists(path_model):
    os.makedirs(path_model)
if not os.path.exists(path_res):
    os.makedirs(path_res)

learning_model_paras(df_train_upper, df_valid_upper, classes, sigma, epochs, batch_size, lr, cost_disType, dis_pars,
                     path_res, path_model, train_name, valid_name, save_root, weights)






