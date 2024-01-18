#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:06:56 2023

@author: LVYE
"""

import tensorflow as tf
import numpy as np
import math, json, os
import io, sqlite3
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error

from utils import extract_batchData_sameMarketSeller
from utils import Convert_to_training_DataFormat
from utils import Compute_unique_mkSelUnit_Pair
from utils import Compute_unique_mkSel_Pair
from utils import Convert_to_training_DataFormat_Class6
from utils import Convert_to_training_DataFormat_Class7
from inferences import CostVector_Inference_withMultiRows
from inferences import CostVector_Inference_withMultiRows_withDict_BuiltIn
from inferences import Sample_Costs_ForLoss1
from inferences import  insert_Costs_into_batchX
from inferences import Insert_AllCosts_3D_Format
from inferences import Correct_logZero_Case
from inferences import Compute_CostVect_Space_Loss3
from inferences import Construct_insertCosts_Format
from inferences import Compute_Loss3_Component
from inferences import compute_rows_costs_Yprobs
from inferences import Compute_Loss3_Component_BuiltIn
from inferences import Insert_AllCosts_3D_Format_Array


def build_rnn_GRUModel(classes, input_dim, step, mask_val):
    # input_dim = 3
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = mask_val, input_shape=(step, input_dim)),
       # tf.keras.layers.BatchNormalization(),
        keras.layers.GRU(5, return_sequences=True),
        #tf.keras.layers.BatchNormalization(),
        keras.layers.GRU(5, return_sequences=False),
       # tf.keras.layers.BatchNormalization(),
        keras.layers.Dense(classes, activation=keras.activations.softmax)
      ])
    model.summary()
    return model

def build_rnn_GRUModel_smaller(classes, input_dim, step, mask_val):
    # input_dim = 3
    model = tf.keras.Sequential([
        keras.layers.Masking(mask_value = mask_val, input_shape=(step, input_dim)),
       # tf.keras.layers.BatchNormalization(),
        keras.layers.GRU(4, return_sequences=True),
        #tf.keras.layers.BatchNormalization(),
        keras.layers.GRU(4, return_sequences=False),
       # tf.keras.layers.BatchNormalization(),
        keras.layers.Dense(classes, activation=keras.activations.softmax)
      ])
    model.summary()
    return model

def Compute_L2Squared_Distance(PredArray, TrueArray):
    
    diff_square = np.power(PredArray - TrueArray, 2)
    distance  = np.sum(diff_square)
    
    return distance

def Compute_Error_CostVec(X_Cost, CostsArray_Inferred, df_CostVect_ID):
    ErrorSet = []
    VisitedTuple = []
    for i in range(df_CostVect_ID.shape[0]):
        
        mk = df_CostVect_ID.loc[i, 'MarketID_period']
        sel = df_CostVect_ID.loc[i, 'userid_profile']
       
        if (mk, sel) not in VisitedTuple:    
            VisitedTuple.append((mk, sel))     
            df_CostVect_ID_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)&(df_CostVect_ID['userid_profile'] == sel)]       
            
            UnitsID_Set = list(df_CostVect_ID_mkSel['unit']) # list.index() return the first index of an element
            IdxVec_Set = [UnitsID_Set.index(ele) for ele in set(UnitsID_Set)] # return a = [1,1,2,2,2,3] => [0, 2, 5]
            
            Units_idxSet = df_CostVect_ID_mkSel.index.tolist()
            PredArray_CostSet_mk_sel = CostsArray_Inferred[Units_idxSet,:]
            TrueArray_CostSet_mk_sel = X_Cost[Units_idxSet,:]
            
            PredArray_costVec_mk_sel = PredArray_CostSet_mk_sel[IdxVec_Set]
            TrueArray_costVec_mk_sel= TrueArray_CostSet_mk_sel[IdxVec_Set]
            
            assert len(PredArray_costVec_mk_sel) == len(np.unique(UnitsID_Set))
            
            ## compute the L2-Squared distance between the pred cost vector and the ture cost vector
            # i.e., d =\sum_{i}(xi-yi)^2
            error_mk_sel = Compute_L2Squared_Distance(PredArray_costVec_mk_sel, TrueArray_costVec_mk_sel)
            ErrorSet.append(error_mk_sel)
    
    Error = np.average(ErrorSet)
    
    return Error

def accuracy_compute(y, logits):
        
    actual_label = np.argmax(y, axis = 1) # find the maximum in each row
    pred_label = np.argmax(logits, axis = 1)
    
    acc = len(np.where(actual_label == pred_label)[0]) / len(actual_label)
    
    return acc

# df_valid_new, classes, batch_size, cost_disType, dis_pars, model, path_res
def Performance_on_ValidData(df_valid, priorDis_dict_valid, classes, bt_size, cost_disType, dis_pars, model, path_res, timeStep):
    # timeStep = 3
    valid_loss, valid_acc, valid_costMse= [], [], []
    valid_epoch_log = {'batch': [] , 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)     
    for batch in range(len(batch_dataFrame)):        
        batch_idataFrame = batch_dataFrame[batch]

        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat(batch_idataFrame, classes) 
        Y_valid = to_categorical(Y_valid, classes)   
        X_valid = X_valid.astype(np.float64)
        
        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows(model, X_valid, Y_valid, X_valid_Cost_UB, df_CostVect_ID, cost_disType, dis_pars)
        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)
        
        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)
        ##### Ensure that c1<c2<c3< ...  in our data: need to reprocess the data? Otherwise, error will be larger  #####
        
        ###################### loss computation ######################        
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)        

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype = float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype = float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)

        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE) 
        logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))             

        loss_1 = cce(y_tensorBatch, logits)
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))
        ############################################  loss-2: KL divergence loss computation ############################################
        dim_EachStep = int(X_valid.shape[1]/timeStep)
        mkSelUnit_uniqueSet = Compute_unique_mkSelUnit_Pair(df_CostVect_ID)
        loss2_Print = []
        for mk, sel, unit in mkSelUnit_uniqueSet:
            df_mkSelUnit = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)&(df_CostVect_ID['userid_profile'] == sel)&(df_CostVect_ID['unit'] == unit)]
            mkSelUnit_idxSet = df_mkSelUnit.index.tolist()
            
            Cost_UB_mkSelUnit_Set = X_valid_Cost_UB[mkSelUnit_idxSet]  
            Cost_UB_mkSelUnit_Unique = list(np.unique(Cost_UB_mkSelUnit_Set)) 
            assert len(Cost_UB_mkSelUnit_Unique) == 1
            Cost_UB_mkSelUnit = Cost_UB_mkSelUnit_Unique[0]
            
            cost_u_UB = np.min([Cost_UB_mkSelUnit, dis_pars['price_max']])
            cost_u = [i for i in range(1, int(cost_u_UB) + 1, dis_pars['gap'])]            
            cost_uInsert_Set = []
            for cost in cost_u: 
                costsInsert = [cost] * len(mkSelUnit_idxSet)
                cost_uInsert_Set.append(costsInsert)

            X_mk_sel_unit = X_valid[mkSelUnit_idxSet, :]
            Y_mk_sel_unit = Y_valid[mkSelUnit_idxSet, :]
            Y_mk_sel_unit_trueIdx = list(np.argmax(Y_mk_sel_unit, axis = 1))

            x_extends_Tensor = Insert_AllCosts_3D_Format(cost_uInsert_Set, X_mk_sel_unit, dim_EachStep)
            Y_result_Tensor = None
            for r in range(x_extends_Tensor.get_shape()[0]):
                Yr_logits_Tensor = model(tf.reshape(x_extends_Tensor[r, :, :], [x_extends_Tensor.get_shape()[1], 3, -1]))
                Yr_probTensor = tf.gather(Yr_logits_Tensor, Y_mk_sel_unit_trueIdx[r], axis = 1)
                Yr_probTensor = tf.reshape(Yr_probTensor, [-1, 1])
                assert Yr_probTensor.get_shape()[0] == x_extends_Tensor.get_shape()[1]
                if r == 0:
                    Y_result_Tensor = Yr_probTensor
                else:
                    Y_result_Tensor = tf.concat([Y_result_Tensor, Yr_probTensor], axis = 1)
            
            ########  try the version of Gfun as follows, i.e, a~np.exp(np.log(a))   ########
            Y_result_Tensor_log = tf.math.log(Y_result_Tensor)
            comp_2nd_temp = tf.math.reduce_sum(Y_result_Tensor_log, axis = 1) # axis=1: add the elements from all columns
            comp_2nd_temp = tf.reshape(comp_2nd_temp, [comp_2nd_temp.get_shape()[0], 1])
            Gfun_c_Arr = tf.math.exp(comp_2nd_temp).numpy()
     
            # prior distribution priorDis_dict
            priorDis_mkSelUnit = priorDis_dict_valid[str((mk, sel, unit))]
            priorDis_mkSelUnit_realSet = []
            for cost in cost_u:
                cost_u_idx = np.where(priorDis_mkSelUnit[:,0] == cost)[0].tolist()[0]
                priorDis_mkSelUnit_realSet.append(priorDis_mkSelUnit[cost_u_idx, 1])
            priorDis_mkSelUnit_realArr = np.array(priorDis_mkSelUnit_realSet).reshape(len(priorDis_mkSelUnit_realSet),1)
    
            ## printing KL-divergence loss (for justifying if it is converged ...) ##
            entropy_prior = np.sum(priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
            entropy_cross = np.sum(priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr/np.sum(Gfun_c_Arr))))
            loss2_Print.append(entropy_prior - entropy_cross)
        
        ######################  loss-3: Constraint loss computation ######################
        # loss3_mkSel_Tensor = tf.convert_to_tensor([0], dtype=float)
        loss3_Print = []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)&(df_CostVect_ID['userid_profile'] == sel)]    
            mkSel_idxSet = df_mkSel.index.tolist() 
            Costs_UB_mkSel = X_valid_Cost_UB[mkSel_idxSet]
            
            UnitsID_Set = list(df_mkSel['unit'])
            IdxVec_Set = [UnitsID_Set.index(ele) for ele in set(UnitsID_Set)] # return a = [1,1,2,2,2,3] => [0, 2, 5]
            Cost_UB_mkSel_eachUnit = Costs_UB_mkSel[IdxVec_Set]
            
            unit_num = len(np.unique(UnitsID_Set)) 
            assert unit_num == len(Cost_UB_mkSel_eachUnit)                
            
            costVec_Space, costVec_Space_Cons = Compute_CostVect_Space_Loss3(unit_num, Cost_UB_mkSel_eachUnit, dis_pars)                        
            
            UnitsID_Array = np.array(UnitsID_Set)                
            costArray_InsertSet = Construct_insertCosts_Format(costVec_Space, UnitsID_Array) 
            costArray_InsertSet_Cons = Construct_insertCosts_Format(costVec_Space_Cons, UnitsID_Array)
            
            X_mk_sel = X_valid[mkSel_idxSet, :]
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis = 1))
    
            x_extends_TensorAll = Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep)       
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component(x_extends_TensorAll, Y_mk_sel_trueIdx, model)                
            x_extends_TensorCons = Insert_AllCosts_3D_Format(costArray_InsertSet_Cons, X_mk_sel, dim_EachStep)       
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component(x_extends_TensorCons, Y_mk_sel_trueIdx, model)
            
            ##### loss3 Print ######
            Cons_loss =  - math.log(np.sum(Efun_c_Cons)) + math.log(np.sum(Efun_c_Arr))
            loss3_Print.append(Cons_loss)
        
            # Grad_BatchSum = tf.reduce_sum(loss_1) + loss2_mkSelUnit_Tensor + loss3_mkSel_Tensor
            #### Correct the Loss2 Computation, since the minimum should be market_seller not market_seller_unit
            mkSelUnit_uniqueArr = np.array(mkSelUnit_uniqueSet)
            assert len(loss2_Print) == len(mkSelUnit_uniqueSet)
            loss2_Print_Real = []
            for mk, sel in mkSel_uniqueSet:
                mkSel_idxSet = np.where((mkSelUnit_uniqueArr[:, 0] == mk)&(mkSelUnit_uniqueArr[:, 1] == sel))[0].tolist()                
                loss_print_mkSel = []
                for idx in mkSel_idxSet:
                    loss_print_mkSel.append(loss2_Print[idx])              
                loss2_Print_Real.append(np.sum(loss_print_mkSel))          
            assert len(loss2_Print_Real) == len(mkSel_uniqueSet)
            
            Print_loss_mean = tf.reduce_mean(loss_1).numpy() + np.average(loss2_Print_Real) + np.average(loss3_Print)
        
        valid_costMse.append(batch_costVec_AvgError)# type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)      
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)  
        valid_epoch_log['acc'].append(acc_batch)
        
    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)    
    
    return valid_loss_mean, valid_acc_mean, valid_costMse_mean


def Performance_on_ValidData_SpeedUp(df_valid, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons,
                                     dict_loss2_validData, dict_loss2_validData_priorDisArr, weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    valid_loss, valid_acc, valid_costMse = [], [], []
    valid_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting!!
        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat(
            batch_idataFrame, classes)
        Y_valid = to_categorical(Y_valid, classes)
        X_valid = X_valid.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_valid, Y_valid,
                                                                                                df_CostVect_ID, cost_disType,
                                                                                                Path_dict_loss3_validData_all, timeStep)

        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        # batch_costVec_AvgError = Compute_Error_CostVec(X_valid_Cost, CostsArray_Inferred, df_CostVect_ID)
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)

        ###################### loss computation ######################
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)
        # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([18, 27], dtype=int32)>
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        loss_1 = cce(y_tensorBatch, logits)  # loss_1.get_shape(): TensorShape([18])
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        ############################################  loss-3 and loss-2 computation ############################################
        dim_EachStep = int(X_valid.shape[1] / timeStep)
        loss3_Print, loss2_Print = [], []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[
                (df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
            mkSel_idxSet = df_mkSel.index.tolist()

            mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
            with open(Path_dict_loss3_validData_all + mkSel_name + '.json', 'r') as f:
                costArray_InsertSet = json.load(f)

            with open(Path_dict_loss3_validData_cons + mkSel_name + '.json', 'r') as fn:
                costArray_InsertSet_Cons = json.load(fn)


            X_mk_sel = X_valid[mkSel_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_TensorAll = Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep)
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model)
            x_extends_TensorCons = Insert_AllCosts_3D_Format(costArray_InsertSet_Cons, X_mk_sel, dim_EachStep)
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,
                                                                           model)

            ##### loss3 Print ######
            Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(max([np.sum(Efun_c_Arr), 1.0e-10]))
            loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

            ######################################### loss2 Set and loss2 Print #################################
            for unit in list(np.unique(df_mkSel['unit'])):
                df_mkSelUnit = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk) &
                                                  (df_CostVect_ID['userid_profile'] == sel) &
                                                  (df_CostVect_ID['unit'] == unit)]
                mkSelUnit_idxSet = df_mkSelUnit.index.tolist()

                cost_uInsert_Set = dict_loss2_validData[str((mk, sel, unit))]

                # len(cost_uInsert_Set): 59
                X_mk_sel_unit = X_valid[mkSelUnit_idxSet, :]  # shape: 4*24
                Y_mk_sel_unit = Y_valid[mkSelUnit_idxSet, :]
                Y_mk_sel_unit_trueIdx = list(np.argmax(Y_mk_sel_unit, axis=1))

                x_extends_Tensor = Insert_AllCosts_3D_Format(cost_uInsert_Set, X_mk_sel_unit, dim_EachStep)
                ## the important component for speeding up ##
                Y_result_Tensor = compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx)

                ######## !! try the version of Gfun as follows, i.e, a~np.exp(np.log(a))  !! ########
                Y_result_Tensor_log = tf.math.log(Y_result_Tensor)  # TensorShape([59, 3])
                comp_2nd_temp = tf.math.reduce_sum(Y_result_Tensor_log,
                                                   axis=1)  # shape: [59], axis=1: add the elements from all columns
                comp_2nd_temp = tf.reshape(comp_2nd_temp, [comp_2nd_temp.get_shape()[0], 1])
                # print('*********************************************************************************')
                Gfun_c_Tensor = tf.math.exp(comp_2nd_temp)
                Gfun_c_Tensor = tf.clip_by_value(Gfun_c_Tensor, 1e-10, 1)

                # print('*********************************************************************************')
                Gfun_c_Arr = Gfun_c_Tensor.numpy()  # tf.math.exp(comp_2nd_temp).numpy() # TensorShape([59])
                priorDis_mkSelUnit_realArr = np.array(dict_loss2_validData_priorDisArr[str((mk, sel, unit))])

                entropy_prior = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                entropy_cross = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr))))
                loss2_Print = loss2_Print + [entropy_prior - entropy_cross]  # * df_mkSelUnit.shape[0]

        Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)

        valid_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)
        valid_epoch_log['acc'].append(acc_batch)

    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)

    return valid_loss_mean, valid_acc_mean, valid_costMse_mean

def Performance_on_ValidData_SpeedUp_Ylabel(df_valid, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons,
                                     dict_loss2_validData, dict_loss2_validData_priorDisArr, weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    valid_loss, valid_acc, valid_costMse = [], [], []
    valid_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting!!
        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat_Class7(
            batch_idataFrame, classes)
        Y_valid = to_categorical(Y_valid, classes)
        X_valid = X_valid.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_valid, Y_valid,
                                                                                                df_CostVect_ID, cost_disType,
                                                                                                Path_dict_loss3_validData_all, timeStep)

        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        # batch_costVec_AvgError = Compute_Error_CostVec(X_valid_Cost, CostsArray_Inferred, df_CostVect_ID)
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)

        ###################### loss computation ######################
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)
        # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([18, 27], dtype=int32)>
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        loss_1 = cce(y_tensorBatch, logits)  # loss_1.get_shape(): TensorShape([18])
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        ############################################  loss-3 and loss-2 computation ############################################
        dim_EachStep = int(X_valid.shape[1] / timeStep)
        loss3_Print, loss2_Print = [], []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[
                (df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
            mkSel_idxSet = df_mkSel.index.tolist()

            mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
            with open(Path_dict_loss3_validData_all + mkSel_name + '.json', 'r') as f:
                costArray_InsertSet = json.load(f)

            with open(Path_dict_loss3_validData_cons + mkSel_name + '.json', 'r') as fn:
                costArray_InsertSet_Cons = json.load(fn)


            X_mk_sel = X_valid[mkSel_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_TensorAll = Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep)
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model)
            x_extends_TensorCons = Insert_AllCosts_3D_Format(costArray_InsertSet_Cons, X_mk_sel, dim_EachStep)
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,
                                                                           model)

            ##### loss3 Print ######
            Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(max([np.sum(Efun_c_Arr), 1.0e-10]))
            loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

            ######################################### loss2 Set and loss2 Print #################################
            for unit in list(np.unique(df_mkSel['unit'])):
                df_mkSelUnit = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk) &
                                                  (df_CostVect_ID['userid_profile'] == sel) &
                                                  (df_CostVect_ID['unit'] == unit)]
                mkSelUnit_idxSet = df_mkSelUnit.index.tolist()

                cost_uInsert_Set = dict_loss2_validData[str((mk, sel, unit))]

                # len(cost_uInsert_Set): 59
                X_mk_sel_unit = X_valid[mkSelUnit_idxSet, :]  # shape: 4*24
                Y_mk_sel_unit = Y_valid[mkSelUnit_idxSet, :]
                Y_mk_sel_unit_trueIdx = list(np.argmax(Y_mk_sel_unit, axis=1))

                x_extends_Tensor = Insert_AllCosts_3D_Format(cost_uInsert_Set, X_mk_sel_unit, dim_EachStep)
                ## the important component for speeding up ##
                Y_result_Tensor = compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx)

                ######## !! try the version of Gfun as follows, i.e, a~np.exp(np.log(a))  !! ########
                Y_result_Tensor_log = tf.math.log(Y_result_Tensor)  # TensorShape([59, 3])
                comp_2nd_temp = tf.math.reduce_sum(Y_result_Tensor_log,
                                                   axis=1)  # shape: [59], axis=1: add the elements from all columns
                comp_2nd_temp = tf.reshape(comp_2nd_temp, [comp_2nd_temp.get_shape()[0], 1])
                # print('*********************************************************************************')
                Gfun_c_Tensor = tf.math.exp(comp_2nd_temp)
                Gfun_c_Tensor = tf.clip_by_value(Gfun_c_Tensor, 1e-10, 1)

                # print('*********************************************************************************')
                Gfun_c_Arr = Gfun_c_Tensor.numpy()  # tf.math.exp(comp_2nd_temp).numpy() # TensorShape([59])
                priorDis_mkSelUnit_realArr = np.array(dict_loss2_validData_priorDisArr[str((mk, sel, unit))])

                entropy_prior = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                entropy_cross = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr))))
                loss2_Print = loss2_Print + [entropy_prior - entropy_cross]  # * df_mkSelUnit.shape[0]

        Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)

        valid_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)
        valid_epoch_log['acc'].append(acc_batch)

    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)

    return valid_loss_mean, valid_acc_mean, valid_costMse_mean

def Callback_EarlyStopping(LossList, min_delta = 0.005, patience = 30):
    # no early stoping for 2*patience epochs
    if len(LossList) // patience < 2:
        return False
    # mean loss for last patience epochs and second-last patience epochs:
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) # second-last
    mean_recent = np.mean(LossList[::-1][:patience]) # last
    delta_abs = np.abs(mean_recent - mean_previous)
    delta_abs = np.abs(delta_abs / mean_previous) # relative change
    if delta_abs < min_delta:
        print('Loss did not change much from last {} epochs '.format(patience), end ='\n')
        print(' Percent change in loss value: {:.4f}% '.format(delta_abs*1e2), end = '\n')
        return True
    else:
        return False


def save_model(model, path_model):
    model_json = model.to_json()
    with open(path_model + 'model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(path_model + 'model.h5')

def load_model(path_model):
    model_weights = path_model + 'model.h5'
    json_file = open(path_model + 'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    return model

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

def Performance_on_ValidData_SpeedUp_v1(df_valid, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons,
                                     dict_loss2_validData, dict_loss2_validData_priorDisArr, weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    valid_loss, valid_acc, valid_costMse = [], [], []
    valid_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting!!
        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat(
            batch_idataFrame, classes)
        Y_valid = to_categorical(Y_valid, classes)
        X_valid = X_valid.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_valid, Y_valid,
                                                                                                df_CostVect_ID, cost_disType,
                                                                                                Path_dict_loss3_validData_all, timeStep)

        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        # batch_costVec_AvgError = Compute_Error_CostVec(X_valid_Cost, CostsArray_Inferred, df_CostVect_ID)
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)

        ###################### loss computation ######################
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)
        # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([18, 27], dtype=int32)>
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        loss_1 = cce(y_tensorBatch, logits)  # loss_1.get_shape(): TensorShape([18])
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        ############################################  loss-3 and loss-2 computation ############################################
        dim_EachStep = int(X_valid.shape[1] / timeStep)
        loss3_Print, loss2_Print = [], []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[
                (df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
            mkSel_idxSet = df_mkSel.index.tolist()

            mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
            with open(Path_dict_loss3_validData_all + mkSel_name + '.json', 'r') as f:
                costArray_InsertSet = json.load(f)

            with open(Path_dict_loss3_validData_cons + mkSel_name + '.json', 'r') as fn:
                costArray_InsertSet_Cons = json.load(fn)


            X_mk_sel = X_valid[mkSel_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_TensorAll = Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep)
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model)
            x_extends_TensorCons = Insert_AllCosts_3D_Format(costArray_InsertSet_Cons, X_mk_sel, dim_EachStep)
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,
                                                                           model)

            ##### loss3 Print ######
            Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(max([np.sum(Efun_c_Arr), 1.0e-10]))
            loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

            ######################################### loss2 Set and loss2 Print #################################
            df_mkSel_reIdx = df_mkSel.reset_index(drop=True)
            UnitsID_SetUnique = list(np.unique(df_mkSel_reIdx['unit']))  ### since not use list(set()), so the order is kept!
            mkSelUnit_idxSet_Set = [df_mkSel_reIdx.loc[df_mkSel_reIdx['unit'] == unit].index.tolist() for unit in UnitsID_SetUnique]
            ###### the inqury takes too long time?

            X_mkSelunit_set = [X_mk_sel[unitIdx_set, :] for unitIdx_set in mkSelUnit_idxSet_Set]
            Y_mk_sel_unit_trueIdx_set = [list(np.array(Y_mk_sel_trueIdx)[unitIdx_set]) for unitIdx_set in
                                         mkSelUnit_idxSet_Set]

            UnitCosts_InsertSet = [dict_loss2_validData[str((mk, sel, unit))] for unit in UnitsID_SetUnique]
            priorDis_mkSelUnit_realArr_set = [np.array(dict_loss2_validData_priorDisArr[str((mk, sel, unit))]) for unit
                                              in UnitsID_SetUnique]
            assert len(UnitCosts_InsertSet) == len(
                UnitsID_SetUnique)  ## the insert cost vector of each unit can be different!

            x_extends_Tensor_set = [Insert_AllCosts_3D_Format(cost_uInsert_Set, X_mk_sel_unit, dim_EachStep) for
                                    cost_uInsert_Set, X_mk_sel_unit in zip(UnitCosts_InsertSet, X_mkSelunit_set)]
            Y_result_Tensor_set = [compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx) for
                                   x_extends_Tensor, Y_mk_sel_unit_trueIdx in
                                   zip(x_extends_Tensor_set, Y_mk_sel_unit_trueIdx_set)]

            def compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr):  # Y_result_Tensor.get_shape(): TensorShape([42,2]); priorDis_mkSelUnit_realArr: (42,1)
                Y_result_Tensor = tf.clip_by_value(Y_result_Tensor, 1e-10, 1)

                # TensorShape([42, 2])
                Gfun_c_Arr = Y_result_Tensor.numpy()  # not log formula but Pr(...) shape, (42, 2)
                entropy_prior = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                entropy_prior_units = entropy_prior * Y_result_Tensor.get_shape()[
                    1]  # the latter is the row number for this unit
                entropy_cross = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr, axis=0))))
                return  entropy_prior_units - entropy_cross

            loss2_Print = [compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr) for Y_result_Tensor, priorDis_mkSelUnit_realArr in zip(Y_result_Tensor_set, priorDis_mkSelUnit_realArr_set)]

        Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)

        valid_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)
        valid_epoch_log['acc'].append(acc_batch)

    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)

    return valid_loss_mean, valid_acc_mean, valid_costMse_mean

def Performance_on_ValidData_SpeedUp_v1_Ylabel(df_valid, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons,
                                     dict_loss2_validData, dict_loss2_validData_priorDisArr, weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    valid_loss, valid_acc, valid_costMse = [], [], []
    valid_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting!!
        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat_Class6(
            batch_idataFrame, classes)
        Y_valid = to_categorical(Y_valid, classes)
        X_valid = X_valid.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_valid, Y_valid,
                                                                                                df_CostVect_ID, cost_disType,
                                                                                                Path_dict_loss3_validData_all, timeStep)

        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        # batch_costVec_AvgError = Compute_Error_CostVec(X_valid_Cost, CostsArray_Inferred, df_CostVect_ID)
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)

        ###################### loss computation ######################
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)
        # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([18, 27], dtype=int32)>
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        loss_1 = cce(y_tensorBatch, logits)  # loss_1.get_shape(): TensorShape([18])
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        ############################################  loss-3 and loss-2 computation ############################################
        dim_EachStep = int(X_valid.shape[1] / timeStep)
        loss3_Print, loss2_Print = [], []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[
                (df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
            mkSel_idxSet = df_mkSel.index.tolist()

            mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
            with open(Path_dict_loss3_validData_all + mkSel_name + '.json', 'r') as f:
                costArray_InsertSet = json.load(f)

            with open(Path_dict_loss3_validData_cons + mkSel_name + '.json', 'r') as fn:
                costArray_InsertSet_Cons = json.load(fn)


            X_mk_sel = X_valid[mkSel_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_TensorAll = Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep)
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model)
            x_extends_TensorCons = Insert_AllCosts_3D_Format(costArray_InsertSet_Cons, X_mk_sel, dim_EachStep)
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,
                                                                           model)

            ##### loss3 Print ######
            Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(max([np.sum(Efun_c_Arr), 1.0e-10]))
            loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

            ######################################### loss2 Set and loss2 Print #################################
            df_mkSel_reIdx = df_mkSel.reset_index(drop=True)
            UnitsID_SetUnique = list(np.unique(df_mkSel_reIdx['unit']))  ### since not use list(set()), so the order is kept!
            mkSelUnit_idxSet_Set = [df_mkSel_reIdx.loc[df_mkSel_reIdx['unit'] == unit].index.tolist() for unit in UnitsID_SetUnique]
            ###### the inqury takes too long time?

            X_mkSelunit_set = [X_mk_sel[unitIdx_set, :] for unitIdx_set in mkSelUnit_idxSet_Set]
            Y_mk_sel_unit_trueIdx_set = [list(np.array(Y_mk_sel_trueIdx)[unitIdx_set]) for unitIdx_set in
                                         mkSelUnit_idxSet_Set]

            UnitCosts_InsertSet = [dict_loss2_validData[str((mk, sel, unit))] for unit in UnitsID_SetUnique]
            priorDis_mkSelUnit_realArr_set = [np.array(dict_loss2_validData_priorDisArr[str((mk, sel, unit))]) for unit
                                              in UnitsID_SetUnique]
            assert len(UnitCosts_InsertSet) == len(
                UnitsID_SetUnique)  ## the insert cost vector of each unit can be different!

            x_extends_Tensor_set = [Insert_AllCosts_3D_Format(cost_uInsert_Set, X_mk_sel_unit, dim_EachStep) for
                                    cost_uInsert_Set, X_mk_sel_unit in zip(UnitCosts_InsertSet, X_mkSelunit_set)]
            Y_result_Tensor_set = [compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx) for
                                   x_extends_Tensor, Y_mk_sel_unit_trueIdx in
                                   zip(x_extends_Tensor_set, Y_mk_sel_unit_trueIdx_set)]

            def compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr):  # Y_result_Tensor.get_shape(): TensorShape([42,2]); priorDis_mkSelUnit_realArr: (42,1)
                Y_result_Tensor = tf.clip_by_value(Y_result_Tensor, 1e-10, 1)

                # TensorShape([42, 2])
                Gfun_c_Arr = Y_result_Tensor.numpy()  # not log formula but Pr(...) shape, (42, 2)
                entropy_prior = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                entropy_prior_units = entropy_prior * Y_result_Tensor.get_shape()[
                    1]  # the latter is the row number for this unit
                entropy_cross = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr, axis=0))))
                return  entropy_prior_units - entropy_cross

            loss2_Print = [compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr) for Y_result_Tensor, priorDis_mkSelUnit_realArr in zip(Y_result_Tensor_set, priorDis_mkSelUnit_realArr_set)]

        Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)

        valid_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)
        valid_epoch_log['acc'].append(acc_batch)

    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)

    return valid_loss_mean, valid_acc_mean, valid_costMse_mean

def Performance_on_ValidData_SpeedUp_v1_sqlite(df_valid, classes, bt_size, cost_disType, dis_pars, model, timeStep,
                                               loss3_all_name_valid,
                                               loss3_cons_name_valid,
                                               loss2_name_valid,
                                               loss2_priorDisArr_name_valid,
                                               weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    valid_loss, valid_acc, valid_costMse = [], [], []
    valid_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)

    conn_loss2 = sqlite3.connect(loss2_name_valid, detect_types=sqlite3.PARSE_DECLTYPES)
    cur_loss2 = conn_loss2.cursor()
    conn_loss2_prior = sqlite3.connect(loss2_priorDisArr_name_valid, detect_types=sqlite3.PARSE_DECLTYPES)
    cur_loss2_prior = conn_loss2_prior.cursor()
    conn_loss3_all = sqlite3.connect(loss3_all_name_valid, detect_types = sqlite3.PARSE_DECLTYPES)
    cur_loss3_all = conn_loss3_all.cursor()
    conn_loss3_cons = sqlite3.connect(loss3_cons_name_valid, detect_types = sqlite3.PARSE_DECLTYPES)
    cur_loss3_cons = conn_loss3_cons.cursor()

    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting!!
        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat(
            batch_idataFrame, classes)
        Y_valid = to_categorical(Y_valid, classes)
        X_valid = X_valid.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_valid, Y_valid,
                                                                                                df_CostVect_ID, cost_disType,
                                                                                                loss3_all_name_valid, timeStep)

        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        # batch_costVec_AvgError = Compute_Error_CostVec(X_valid_Cost, CostsArray_Inferred, df_CostVect_ID)
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)

        ###################### loss computation ######################
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)
        # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([18, 27], dtype=int32)>
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        loss_1 = cce(y_tensorBatch, logits)  # loss_1.get_shape(): TensorShape([18])
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        ############################################  loss-3 and loss-2 computation ############################################
        dim_EachStep = int(X_valid.shape[1] / timeStep)
        loss3_Print, loss2_Print = [], []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[
                (df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
            mkSel_idxSet = df_mkSel.index.tolist()

            # mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
            # with open(Path_dict_loss3_validData_all + mkSel_name + '.json', 'r') as f:
            #     costArray_InsertSet = json.load(f)
            # with open(Path_dict_loss3_validData_cons + mkSel_name + '.json', 'r') as fn:
            #     costArray_InsertSet_Cons = json.load(fn)
            cur_loss3_all.execute("SELECT arr FROM loss3_all_valid WHERE market =? AND seller =? ", (mk, sel))
            costArray_InsertArray = cur_loss3_all.fetchone()[0]
            cur_loss3_cons.execute("SELECT arr FROM loss3_cons_valid WHERE market =? AND seller = ? ", (mk, sel))
            costArray_InsertArray_Cons = cur_loss3_cons.fetchone()[0]

            X_mk_sel = X_valid[mkSel_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_TensorAll = Insert_AllCosts_3D_Format_Array(costArray_InsertArray, X_mk_sel, dim_EachStep)
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model)
            x_extends_TensorCons = Insert_AllCosts_3D_Format_Array(costArray_InsertArray_Cons, X_mk_sel, dim_EachStep)
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,
                                                                           model)

            ##### loss3 Print ######
            Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(max([np.sum(Efun_c_Arr), 1.0e-10]))
            loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

            ######################################### loss2 Set and loss2 Print #################################
            df_mkSel_reIdx = df_mkSel.reset_index(drop=True)
            UnitsID_SetUnique = list(np.unique(df_mkSel_reIdx['unit']))  ### since not use list(set()), so the order is kept!
            mkSelUnit_idxSet_Set = [df_mkSel_reIdx.loc[df_mkSel_reIdx['unit'] == unit].index.tolist() for unit in UnitsID_SetUnique]
            ###### the inqury takes too long time?

            X_mkSelunit_set = [X_mk_sel[unitIdx_set, :] for unitIdx_set in mkSelUnit_idxSet_Set]
            Y_mk_sel_unit_trueIdx_set = [list(np.array(Y_mk_sel_trueIdx)[unitIdx_set]) for unitIdx_set in
                                         mkSelUnit_idxSet_Set]

            # UnitCosts_InsertSet = [dict_loss2_validData[str((mk, sel, unit))] for unit in UnitsID_SetUnique]
            # priorDis_mkSelUnit_realArr_set = [np.array(dict_loss2_validData_priorDisArr[str((mk, sel, unit))]) for unit
            #                                   in UnitsID_SetUnique]

            UnitCosts_InsertSet, priorDis_mkSelUnit_realArr_set = [], []
            for unit in UnitsID_SetUnique: ### when use SELECT, remember modify the table name!!!
                cur_loss2.execute(
                    "SELECT arr FROM loss2_valid WHERE market =? AND seller = ? AND unit = ? ", (mk, sel, int(unit)))
                UnitCosts_InsertSet.append(cur_loss2.fetchone()[0])
                cur_loss2_prior.execute(
                    "SELECT arr FROM loss2_prior_valid WHERE market =? AND seller = ? AND unit = ? ",
                    (mk, sel, int(unit)))
                priorDis_mkSelUnit_realArr_set.append(cur_loss2_prior.fetchone()[0])
            assert len(UnitCosts_InsertSet) == len(UnitsID_SetUnique)  ## the insert cost vector of each unit can be different!

            x_extends_Tensor_set = [Insert_AllCosts_3D_Format_Array(cost_uInsert_Array, X_mk_sel_unit, dim_EachStep) for
                                    cost_uInsert_Array, X_mk_sel_unit in zip(UnitCosts_InsertSet, X_mkSelunit_set)]
            Y_result_Tensor_set = [compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx) for
                                   x_extends_Tensor, Y_mk_sel_unit_trueIdx in
                                   zip(x_extends_Tensor_set, Y_mk_sel_unit_trueIdx_set)]

            def compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr):  # Y_result_Tensor.get_shape(): TensorShape([42,2]); priorDis_mkSelUnit_realArr: (42,1)
                Y_result_Tensor = tf.clip_by_value(Y_result_Tensor, 1e-10, 1)

                # TensorShape([42, 2])
                Gfun_c_Arr = Y_result_Tensor.numpy()  # not log formula but Pr(...) shape, (42, 2)
                entropy_prior = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                entropy_prior_units = entropy_prior * Y_result_Tensor.get_shape()[
                    1]  # the latter is the row number for this unit
                entropy_cross = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr, axis=0))))
                return  entropy_prior_units - entropy_cross

            loss2_Print = [compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr) for Y_result_Tensor, priorDis_mkSelUnit_realArr in zip(Y_result_Tensor_set, priorDis_mkSelUnit_realArr_set)]

        Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)

        valid_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)
        valid_epoch_log['acc'].append(acc_batch)

    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)

    return valid_loss_mean, valid_acc_mean, valid_costMse_mean

def Performance_on_ValidData_SpeedUp_v1_sqlite_Ylabel(df_valid, classes, bt_size, cost_disType, dis_pars, model, timeStep,
                                               loss3_all_name_valid,
                                               loss3_cons_name_valid,
                                               loss2_name_valid,
                                               loss2_priorDisArr_name_valid,
                                               weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    valid_loss, valid_acc, valid_costMse = [], [], []
    valid_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)

    conn_loss2 = sqlite3.connect(loss2_name_valid, detect_types=sqlite3.PARSE_DECLTYPES)
    cur_loss2 = conn_loss2.cursor()
    conn_loss2_prior = sqlite3.connect(loss2_priorDisArr_name_valid, detect_types=sqlite3.PARSE_DECLTYPES)
    cur_loss2_prior = conn_loss2_prior.cursor()
    conn_loss3_all = sqlite3.connect(loss3_all_name_valid, detect_types = sqlite3.PARSE_DECLTYPES)
    cur_loss3_all = conn_loss3_all.cursor()
    conn_loss3_cons = sqlite3.connect(loss3_cons_name_valid, detect_types = sqlite3.PARSE_DECLTYPES)
    cur_loss3_cons = conn_loss3_cons.cursor()

    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting!!
        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat_Class6(
            batch_idataFrame, classes)
        Y_valid = to_categorical(Y_valid, classes)
        X_valid = X_valid.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_valid, Y_valid,
                                                                                                df_CostVect_ID, cost_disType,
                                                                                                loss3_all_name_valid, timeStep)

        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        # batch_costVec_AvgError = Compute_Error_CostVec(X_valid_Cost, CostsArray_Inferred, df_CostVect_ID)
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)

        ###################### loss computation ######################
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)
        # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([18, 27], dtype=int32)>
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        loss_1 = cce(y_tensorBatch, logits)  # loss_1.get_shape(): TensorShape([18])
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        ############################################  loss-3 and loss-2 computation ############################################
        dim_EachStep = int(X_valid.shape[1] / timeStep)
        loss3_Print, loss2_Print = [], []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[
                (df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
            mkSel_idxSet = df_mkSel.index.tolist()

            # mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
            # with open(Path_dict_loss3_validData_all + mkSel_name + '.json', 'r') as f:
            #     costArray_InsertSet = json.load(f)
            # with open(Path_dict_loss3_validData_cons + mkSel_name + '.json', 'r') as fn:
            #     costArray_InsertSet_Cons = json.load(fn)
            cur_loss3_all.execute("SELECT arr FROM loss3_all_valid WHERE market =? AND seller =? ", (mk, sel))
            costArray_InsertArray = cur_loss3_all.fetchone()[0]
            cur_loss3_cons.execute("SELECT arr FROM loss3_cons_valid WHERE market =? AND seller = ? ", (mk, sel))
            costArray_InsertArray_Cons = cur_loss3_cons.fetchone()[0]

            X_mk_sel = X_valid[mkSel_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_TensorAll = Insert_AllCosts_3D_Format_Array(costArray_InsertArray, X_mk_sel, dim_EachStep)
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model)
            x_extends_TensorCons = Insert_AllCosts_3D_Format_Array(costArray_InsertArray_Cons, X_mk_sel, dim_EachStep)
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,
                                                                           model)

            ##### loss3 Print ######
            Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(max([np.sum(Efun_c_Arr), 1.0e-10]))
            loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

            ######################################### loss2 Set and loss2 Print #################################
            df_mkSel_reIdx = df_mkSel.reset_index(drop=True)
            UnitsID_SetUnique = list(np.unique(df_mkSel_reIdx['unit']))  ### since not use list(set()), so the order is kept!
            mkSelUnit_idxSet_Set = [df_mkSel_reIdx.loc[df_mkSel_reIdx['unit'] == unit].index.tolist() for unit in UnitsID_SetUnique]
            ###### the inqury takes too long time?

            X_mkSelunit_set = [X_mk_sel[unitIdx_set, :] for unitIdx_set in mkSelUnit_idxSet_Set]
            Y_mk_sel_unit_trueIdx_set = [list(np.array(Y_mk_sel_trueIdx)[unitIdx_set]) for unitIdx_set in
                                         mkSelUnit_idxSet_Set]

            # UnitCosts_InsertSet = [dict_loss2_validData[str((mk, sel, unit))] for unit in UnitsID_SetUnique]
            # priorDis_mkSelUnit_realArr_set = [np.array(dict_loss2_validData_priorDisArr[str((mk, sel, unit))]) for unit
            #                                   in UnitsID_SetUnique]

            UnitCosts_InsertSet, priorDis_mkSelUnit_realArr_set = [], []
            for unit in UnitsID_SetUnique: ### when use SELECT, remember modify the table name!!!
                cur_loss2.execute(
                    "SELECT arr FROM loss2_valid WHERE market =? AND seller = ? AND unit = ? ", (mk, sel, int(unit)))
                UnitCosts_InsertSet.append(cur_loss2.fetchone()[0])
                cur_loss2_prior.execute(
                    "SELECT arr FROM loss2_prior_valid WHERE market =? AND seller = ? AND unit = ? ",
                    (mk, sel, int(unit)))
                priorDis_mkSelUnit_realArr_set.append(cur_loss2_prior.fetchone()[0])
            assert len(UnitCosts_InsertSet) == len(UnitsID_SetUnique)  ## the insert cost vector of each unit can be different!

            x_extends_Tensor_set = [Insert_AllCosts_3D_Format_Array(cost_uInsert_Array, X_mk_sel_unit, dim_EachStep) for
                                    cost_uInsert_Array, X_mk_sel_unit in zip(UnitCosts_InsertSet, X_mkSelunit_set)]
            Y_result_Tensor_set = [compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx) for
                                   x_extends_Tensor, Y_mk_sel_unit_trueIdx in
                                   zip(x_extends_Tensor_set, Y_mk_sel_unit_trueIdx_set)]

            def compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr):  # Y_result_Tensor.get_shape(): TensorShape([42,2]); priorDis_mkSelUnit_realArr: (42,1)
                Y_result_Tensor = tf.clip_by_value(Y_result_Tensor, 1e-10, 1)

                # TensorShape([42, 2])
                Gfun_c_Arr = Y_result_Tensor.numpy()  # not log formula but Pr(...) shape, (42, 2)
                entropy_prior = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                entropy_prior_units = entropy_prior * Y_result_Tensor.get_shape()[
                    1]  # the latter is the row number for this unit
                entropy_cross = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr, axis=0))))
                return  entropy_prior_units - entropy_cross

            loss2_Print = [compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr) for Y_result_Tensor, priorDis_mkSelUnit_realArr in zip(Y_result_Tensor_set, priorDis_mkSelUnit_realArr_set)]

        Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)

        valid_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)
        valid_epoch_log['acc'].append(acc_batch)

    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)

    return valid_loss_mean, valid_acc_mean, valid_costMse_mean

def Performance_on_ValidData_SpeedUp_v1_Ylabel2(df_valid, classes, bt_size, cost_disType, dis_pars, model, timeStep, Path_dict_loss3_validData_all, Path_dict_loss3_validData_cons,
                                     dict_loss2_validData, dict_loss2_validData_priorDisArr, weights):
    w1, w2, w3 = weights[0], weights[1], weights[2]
    valid_loss, valid_acc, valid_costMse = [], [], []
    valid_epoch_log = {'batch': [], 'total_loss': [], 'acc': [], 'cost_mse': []}
    batch_dataFrame, batch_pairIdx = extract_batchData_sameMarketSeller(df_valid, bt_size)
    for batch in range(len(batch_dataFrame)):
        batch_idataFrame = batch_dataFrame[batch]
        # convert into a recognizable data format by RNN and watch the Y(labels) setting!!
        X_valid, Y_valid, X_valid_Cost, X_valid_Cost_UB, df_CostVect_ID = Convert_to_training_DataFormat_Class7(
            batch_idataFrame, classes)
        Y_valid = to_categorical(Y_valid, classes)
        X_valid = X_valid.astype(np.float64)

        ## make inference for cost vector given the current model ##
        CostsArray_Inferred, Dict_costSet = CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X_valid, Y_valid,
                                                                                                df_CostVect_ID, cost_disType,
                                                                                                Path_dict_loss3_validData_all, timeStep)

        CostsArray_Inferred = CostsArray_Inferred.reshape(len(CostsArray_Inferred), 1)  # (18,1)

        ####### COMPUTE THE DISTANCE OF TWO VECTORS..... #######
        # batch_costVec_AvgError = Compute_Error_CostVec(X_valid_Cost, CostsArray_Inferred, df_CostVect_ID)
        batch_costVec_AvgError = mean_squared_error(X_valid_Cost, CostsArray_Inferred)

        ###################### loss computation ######################
        CostSampled_BatchArr = Sample_Costs_ForLoss1(X_valid_Cost_UB, df_CostVect_ID, dis_pars)
        CostSampled_BatchArr = CostSampled_BatchArr.reshape(len(CostSampled_BatchArr), 1)

        ############################################  loss-1: cross-entropy computation ############################################
        X_BatchTensor = tf.convert_to_tensor(X_valid, dtype=float)
        y_tensorBatch = tf.convert_to_tensor(Y_valid)
        CostSampled_BatchTensor = tf.convert_to_tensor(CostSampled_BatchArr, dtype=float)
        X_exBatch_Tensor = insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep)
        # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([18, 27], dtype=int32)>
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # logits = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits_org = model(tf.reshape(X_exBatch_Tensor, [X_exBatch_Tensor.get_shape()[0], 3, -1]))
        logits = tf.clip_by_value(logits_org, 1e-10, 1)
        loss_1 = cce(y_tensorBatch, logits)  # loss_1.get_shape(): TensorShape([18])
        acc_batch = tf.convert_to_tensor(accuracy_compute(np.array(y_tensorBatch), logits.numpy()))

        ############################################  loss-3 and loss-2 computation ############################################
        dim_EachStep = int(X_valid.shape[1] / timeStep)
        loss3_Print, loss2_Print = [], []
        mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_CostVect_ID)
        for mk, sel in mkSel_uniqueSet:
            df_mkSel = df_CostVect_ID.loc[
                (df_CostVect_ID['MarketID_period'] == mk) & (df_CostVect_ID['userid_profile'] == sel)]
            mkSel_idxSet = df_mkSel.index.tolist()

            mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # costArray_InsertSet = dict_loss3_trainData_all[str((mk, sel))]
            with open(Path_dict_loss3_validData_all + mkSel_name + '.json', 'r') as f:
                costArray_InsertSet = json.load(f)

            with open(Path_dict_loss3_validData_cons + mkSel_name + '.json', 'r') as fn:
                costArray_InsertSet_Cons = json.load(fn)


            X_mk_sel = X_valid[mkSel_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y_valid[mkSel_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_TensorAll = Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep)
            Grad_TensorAll, Efun_c_Arr = Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model)
            x_extends_TensorCons = Insert_AllCosts_3D_Format(costArray_InsertSet_Cons, X_mk_sel, dim_EachStep)
            Grad_TensorCons, Efun_c_Cons = Compute_Loss3_Component_BuiltIn(x_extends_TensorCons, Y_mk_sel_trueIdx,
                                                                           model)

            ##### loss3 Print ######
            Cons_loss = - math.log(max([np.sum(Efun_c_Cons), 1.0e-10])) + math.log(max([np.sum(Efun_c_Arr), 1.0e-10]))
            loss3_Print = loss3_Print + [Cons_loss]  # * df_mkSel.shape[0]

            ######################################### loss2 Set and loss2 Print #################################
            df_mkSel_reIdx = df_mkSel.reset_index(drop=True)
            UnitsID_SetUnique = list(np.unique(df_mkSel_reIdx['unit']))  ### since not use list(set()), so the order is kept!
            mkSelUnit_idxSet_Set = [df_mkSel_reIdx.loc[df_mkSel_reIdx['unit'] == unit].index.tolist() for unit in UnitsID_SetUnique]
            ###### the inqury takes too long time?

            X_mkSelunit_set = [X_mk_sel[unitIdx_set, :] for unitIdx_set in mkSelUnit_idxSet_Set]
            Y_mk_sel_unit_trueIdx_set = [list(np.array(Y_mk_sel_trueIdx)[unitIdx_set]) for unitIdx_set in
                                         mkSelUnit_idxSet_Set]

            UnitCosts_InsertSet = [dict_loss2_validData[str((mk, sel, unit))] for unit in UnitsID_SetUnique]
            priorDis_mkSelUnit_realArr_set = [np.array(dict_loss2_validData_priorDisArr[str((mk, sel, unit))]) for unit
                                              in UnitsID_SetUnique]
            assert len(UnitCosts_InsertSet) == len(
                UnitsID_SetUnique)  ## the insert cost vector of each unit can be different!

            x_extends_Tensor_set = [Insert_AllCosts_3D_Format(cost_uInsert_Set, X_mk_sel_unit, dim_EachStep) for
                                    cost_uInsert_Set, X_mk_sel_unit in zip(UnitCosts_InsertSet, X_mkSelunit_set)]
            Y_result_Tensor_set = [compute_rows_costs_Yprobs(x_extends_Tensor, model, Y_mk_sel_unit_trueIdx) for
                                   x_extends_Tensor, Y_mk_sel_unit_trueIdx in
                                   zip(x_extends_Tensor_set, Y_mk_sel_unit_trueIdx_set)]

            def compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr):  # Y_result_Tensor.get_shape(): TensorShape([42,2]); priorDis_mkSelUnit_realArr: (42,1)
                Y_result_Tensor = tf.clip_by_value(Y_result_Tensor, 1e-10, 1)

                # TensorShape([42, 2])
                Gfun_c_Arr = Y_result_Tensor.numpy()  # not log formula but Pr(...) shape, (42, 2)
                entropy_prior = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(priorDis_mkSelUnit_realArr)))
                entropy_prior_units = entropy_prior * Y_result_Tensor.get_shape()[
                    1]  # the latter is the row number for this unit
                entropy_cross = np.sum(
                    priorDis_mkSelUnit_realArr * np.log(Correct_logZero_Case(Gfun_c_Arr / np.sum(Gfun_c_Arr, axis=0))))
                return  entropy_prior_units - entropy_cross

            loss2_Print = [compute_loss2_Print(Y_result_Tensor, priorDis_mkSelUnit_realArr) for Y_result_Tensor, priorDis_mkSelUnit_realArr in zip(Y_result_Tensor_set, priorDis_mkSelUnit_realArr_set)]

        Print_loss_mean = w1 * tf.reduce_mean(loss_1).numpy() + w2 * np.mean(loss2_Print) + w3 * np.mean(loss3_Print)

        valid_costMse.append(batch_costVec_AvgError)  # type(batch_costVec_AvgError): numpy.float64
        valid_loss.append(Print_loss_mean)
        valid_acc.append(acc_batch)
        valid_epoch_log['batch'].append(batch + 1)
        valid_epoch_log['cost_mse'].append(batch_costVec_AvgError)
        valid_epoch_log['total_loss'].append(Print_loss_mean)
        valid_epoch_log['acc'].append(acc_batch)

    valid_loss_mean = np.mean(valid_loss)
    valid_acc_mean = np.mean(valid_acc)
    valid_costMse_mean = np.mean(valid_costMse)

    return valid_loss_mean, valid_acc_mean, valid_costMse_mean
