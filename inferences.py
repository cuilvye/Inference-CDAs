#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:05:59 2023

@author: lvye
"""

import numpy as np
import itertools
import copy, random
import json
import tensorflow as tf
import io, sqlite3

def Compute_CostVect_Space(unit_num, UB_sets_costVect, dis_pars):
    gap = dis_pars['gap']
    cost_u_set = []
    for u in range(unit_num):
        cost_u_UB = np.min([UB_sets_costVect[u], dis_pars['price_max']])
        cost_u = [i for i in range(1, int(cost_u_UB) + 1, gap)]
        cost_u_set.append(cost_u)
        
    # costVec_space = list(itertools.product(*cost_u_set))#[(1, 1, 1), (1, 1, 2),(1, 1, 3),...,]
    # costVec_space = list(filter(lambda e: e[1] - e[0] <= 60 and e[2] - e[1] <= 60, itertools.product(*cost_u_set)))
    if unit_num == 1:
        costVec_space = list(itertools.product(*cost_u_set))
    elif unit_num == 2: # since the maximum value of unit number is 3 in our case, so we have,
        costVec_space = list(filter(lambda e: e[1] - e[0] <= 60, itertools.product(*cost_u_set)))
    elif unit_num == 3:
        costVec_space = list(filter(lambda e: e[1] - e[0] <= 60 and e[2] - e[1] <= 60, itertools.product(*cost_u_set)))       
    else:
        print(' unit_num is {}, Do you filter out those rows in our data?'.format(unit_num))
        assert 1 == 0
    return costVec_space

def Construct_insertCosts_Format(costVec_Space, UnitsID_Array):
    # determine the row number of each unit cost:
    UnitsID_Set = list(UnitsID_Array)
    dict_unitNum = {i: UnitsID_Set.count(i) for i in UnitsID_Set}   
    
    CostsInsert_Set = []
    for costVec in costVec_Space: 
        costVec_New = []
        for key in dict_unitNum.keys(): # 1,2,3
            costVec_New = costVec_New + [costVec[int(key)-1]] * dict_unitNum[key]
        assert len(costVec_New) == len(UnitsID_Set)
        CostsInsert_Set.append(costVec_New)
    assert len(CostsInsert_Set) == len(costVec_Space)
    
    return CostsInsert_Set

def is_arithmetic(l):
    delta = l[1] - l[0]
    for index in range(len(l) - 1):
        if not (l[index + 1] - l[index] == delta):
             return False
    return True        

def extractDigits(lst):
    return [[el] for el in lst]

def extractDigits_addID(lst):
    res = []
    for i in range(len(lst)):
        res.append([i, lst[i]])
    return res

def Compute_PosterProb_costSet(x_extends_Tensor, model, Y_mk_sel_trueIdx):
    Y_result_Tensor = None
    for r in range(x_extends_Tensor.get_shape()[0]):
        Yr_logits_Tensor = model(tf.reshape(x_extends_Tensor[r, :, :], [x_extends_Tensor.get_shape()[1], 3, -1]))
        Yr_probTensor = tf.gather(Yr_logits_Tensor, Y_mk_sel_trueIdx[r], axis = 1)
        Yr_probTensor = tf.reshape(Yr_probTensor, [-1, 1])
        assert Yr_probTensor.get_shape()[0] == x_extends_Tensor.get_shape()[1]
        if r == 0:
            Y_result_Tensor = Yr_probTensor
        else:
            Y_result_Tensor = tf.concat([Y_result_Tensor, Yr_probTensor], axis = 1)
    CostSet_ProbTensor = tf.math.reduce_prod(Y_result_Tensor, axis = 1)
    CostSet_PostProbTensor = tf.divide(CostSet_ProbTensor, tf.reduce_sum(CostSet_ProbTensor))
    
    return CostSet_PostProbTensor

def Tensor2d_slice(A, B):
    """ Returns values of rows i of A at column B[i]
    where A is a 2D Tensor with shape [None, D], and B is a 1D Tensor with shape [None] with type int32 elements in [0,D)
    Example:
      A =[[1,2], B = [0,1], vector_slice(A,B) -> [1,4]
          [3,4]]
    """
    B = tf.expand_dims(B, 1)
    range = tf.expand_dims(tf.range(tf.shape(B)[0]), 1)
    ind = tf.concat([range, B], 1)
    return tf.gather_nd(A, ind)

def Compute_PosterProb_costSet_BuiltIn(x_extends_Tensor, model, Y_mk_sel_trueIdx):

    x_extends_Tensor_2d = tf.reshape(x_extends_Tensor, [-1, x_extends_Tensor.get_shape()[-1]])
    x_extends_Tensor_2d_inputModel = tf.reshape(x_extends_Tensor_2d, [x_extends_Tensor_2d.get_shape()[0], 3, -1])

    Y_logits_Tensor_org = model(x_extends_Tensor_2d_inputModel)
    Y_logits_Tensor = tf.clip_by_value(Y_logits_Tensor_org, 1e-10, 1)

    trueIdx_slices = list(np.repeat(Y_mk_sel_trueIdx, x_extends_Tensor.get_shape()[1]))
    Y_result_Tensor_Temp = Tensor2d_slice(Y_logits_Tensor, trueIdx_slices)
    Y_result_Tensor_trans = tf.reshape(Y_result_Tensor_Temp, [x_extends_Tensor.get_shape()[0], -1])
    Y_result_Tensor = tf.transpose(Y_result_Tensor_trans)

    CostSet_ProbTensor = tf.math.reduce_prod(Y_result_Tensor, axis=1)  # the product of each row
    CostSet_PostProbTensor = tf.divide(CostSet_ProbTensor, tf.reduce_sum(CostSet_ProbTensor))

    return CostSet_PostProbTensor


def CostVector_Inference_withMultiRows(model, X, Y, X_Cost_UB, df_CostVect_ID, cost_disType, dis_pars):    
    ##### Since we do not know the true prior distribution of cost vector for a seller, so we consider 
    ##### Uniform type, namely, Pr(c1) Pr(c2)... Uniform & Pr(c1, c2, ...) also Uniform, Thus, the prior
    ##### in Equation (2) all could be eliminated...
    assert cost_disType == 'Uniform'
    timeStep = 3
    dim_EachStep = int(X.shape[1]/timeStep) # dim_EachStep: 8, and 3 is the timeStep, if timeStep is not 3, should modify it. BE CAREFUL!!!
    VisitedTuple = []
    Dict_sel_costSet = {} # Firstly, get the costVec for each seller
    Result_Pred_CostSet = []
    for i in range(df_CostVect_ID.shape[0]): #tqdm(range(df_CostVect_ID.shape[0]))
        mk = df_CostVect_ID.loc[i, 'MarketID_period']
        sel = df_CostVect_ID.loc[i, 'userid_profile']
        if (mk, sel) not in VisitedTuple:    
            VisitedTuple.append((mk, sel))    
            # df_CostVect_ID_mkSel is only used for computing the unit number 
            df_CostVect_ID_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)
                                                      &(df_CostVect_ID['userid_profile'] == sel)]       
            
      
            units_idxSet = df_CostVect_ID_mkSel.index.tolist()   
            if len(units_idxSet) > 1:
                assert is_arithmetic(units_idxSet) == True ##Ensure that rows from one seller is close               
            # c1 < c2, and rows in df, should be row1 < row2
            Cost_UB_mk_sel = X_Cost_UB[units_idxSet]
            
            UnitsID_Set = list(df_CostVect_ID_mkSel['unit']) # set(list) will change the order in  original list
            IdxVec_Set = [UnitsID_Set.index(ele) for ele in list(np.unique(UnitsID_Set))]
            UB_sets_costVec = Cost_UB_mk_sel[IdxVec_Set]
            
            unit_num = len(np.unique(df_CostVect_ID_mkSel['unit'])) 
            assert unit_num == len(UB_sets_costVec)
            
            costVec_Space = Compute_CostVect_Space(unit_num, UB_sets_costVec, dis_pars)            
            # dict_sel_costVec[(mk, sel)] = costVec_Space
        
            UnitsID_Array = np.array(df_CostVect_ID_mkSel['unit'])
            costArray_Insert_Set = Construct_insertCosts_Format(costVec_Space, UnitsID_Array) 
            
            assert len(costVec_Space) == len(costArray_Insert_Set)
            # make inference for the cost vector
            X_mk_sel = X[units_idxSet, :]
            Y_mk_sel = Y[units_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis = 1))

            x_extends_Tensor = Insert_AllCosts_3D_Format(costArray_Insert_Set, X_mk_sel, dim_EachStep)
            CostSet_PostProbTensor = Compute_PosterProb_costSet(x_extends_Tensor, model, Y_mk_sel_trueIdx)
            
            ID_argmax = tf.math.argmax(CostSet_PostProbTensor)                
            Pred_costSet = costArray_Insert_Set[ID_argmax]
            
            Dict_sel_costSet[(mk, sel)] = Pred_costSet
            Result_Pred_CostSet = Result_Pred_CostSet + Pred_costSet

            
    assert len(Result_Pred_CostSet) ==  df_CostVect_ID.shape[0]       
    
    return np.array(Result_Pred_CostSet), Dict_sel_costSet

def CostVector_Inference_withMultiRows_withDict(model, X, Y, X_Cost_UB, df_CostVect_ID, cost_disType, dis_pars, Path_dict_loss3_trainData_all):    
    ##### Since we do not know the true prior distribution of cost vector for a seller, so we consider 
    ##### Uniform type, namely, Pr(c1) Pr(c2)... Uniform & Pr(c1, c2, ...) also Uniform, Thus, the prior
    ##### in Equation (2) all could be eliminated...
    assert cost_disType == 'Uniform'
    timeStep = 3
    dim_EachStep = int(X.shape[1]/timeStep) # dim_EachStep: 8, and 3 is the timeStep, if timeStep is not 3, should modify it. BE CAREFUL!!!
    VisitedTuple = []
    Dict_sel_costSet = {} # Firstly, get the costVec for each seller
    Result_Pred_CostSet = []
    for i in range(df_CostVect_ID.shape[0]): #tqdm(range(df_CostVect_ID.shape[0]))
        
        mk = df_CostVect_ID.loc[i, 'MarketID_period']
        sel = df_CostVect_ID.loc[i, 'userid_profile']
       
        if (mk, sel) not in VisitedTuple:    
            VisitedTuple.append((mk, sel))    
            # df_CostVect_ID_mkSel is only used for computing the unit number 
            df_CostVect_ID_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)
                                                      &(df_CostVect_ID['userid_profile'] == sel)]       
                 
            units_idxSet = df_CostVect_ID_mkSel.index.tolist()   
            if len(units_idxSet) > 1:
                assert is_arithmetic(units_idxSet) == True ##Ensure that rows from one seller is close               

            mkSel_name = '-'.join([str(int(mk)),str(int(sel))])
            with open(Path_dict_loss3_trainData_all + mkSel_name + '.json', 'r') as f:
                costArray_Insert_Set  = json.load(f)
            
            # len(costArray_Insert_Set): 159936
            # make inference for the cost vector
            X_mk_sel = X[units_idxSet, :] #shape: 4*24
            Y_mk_sel = Y[units_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis = 1))
            # Y_mk_sel_trueIdx_indices = extractDigits(Y_mk_sel_trueIdx) # for tf.gather_nd, extract the prob Pr(a_{l}|c,x;\theta)
            # Y_mk_sel_trueIdx_indices = extractDigits_addID(Y_mk_sel_trueIdx)

            x_extends_Tensor = Insert_AllCosts_3D_Format(costArray_Insert_Set, X_mk_sel, dim_EachStep)
            CostSet_PostProbTensor = Compute_PosterProb_costSet(x_extends_Tensor, model, Y_mk_sel_trueIdx)
            ID_argmax = tf.math.argmax(CostSet_PostProbTensor)                
            Pred_costSet = costArray_Insert_Set[ID_argmax]
            
            Dict_sel_costSet[(mk, sel)] = Pred_costSet
            Result_Pred_CostSet = Result_Pred_CostSet + Pred_costSet
            
    assert len(Result_Pred_CostSet) ==  df_CostVect_ID.shape[0]       
    
    return np.array(Result_Pred_CostSet), Dict_sel_costSet


def CostVector_Inference_withMultiRows_withDict_BuiltIn(model, X, Y, df_CostVect_ID, cost_disType, Path_dict_loss3_trainData_all, timeStep):
    ##### Since we do not know the true prior distribution of cost vector for a seller, so we consider
    ##### Uniform type, namely, Pr(c1) Pr(c2)... Uniform & Pr(c1, c2, ...) also Uniform, Thus, the prior
    ##### in Equation (2) all could be eliminated...
    assert cost_disType == 'Uniform'
    # timeStep = 3
    dim_EachStep = int(X.shape[1] / timeStep)  # dim_EachStep: 8, and 3 is the timeStep, if timeStep is not 3, should modify it. BE CAREFUL!!!
    VisitedTuple = []
    Dict_sel_costSet = {}  # Firstly, get the costVec for each seller
    Result_Pred_CostSet = []
    for i in range(df_CostVect_ID.shape[0]):  # tqdm(range(df_CostVect_ID.shape[0]))

        mk = df_CostVect_ID.loc[i, 'MarketID_period']
        sel = df_CostVect_ID.loc[i, 'userid_profile']

        if (mk, sel) not in VisitedTuple:
            VisitedTuple.append((mk, sel))
            # df_CostVect_ID_mkSel is only used for computing the unit number
            df_CostVect_ID_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)
                                                      & (df_CostVect_ID['userid_profile'] == sel)]

            units_idxSet = df_CostVect_ID_mkSel.index.tolist()
            if len(units_idxSet) > 1:
                assert is_arithmetic(units_idxSet) == True  ##Ensure that rows from one seller is close

            # assert len(costVec_Space) == len(costArray_Insert_Set)
            mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            with open(Path_dict_loss3_trainData_all + mkSel_name + '.json', 'r') as f:
                costArray_Insert_Set = json.load(f)

            # len(costArray_Insert_Set): 159936
            # make inference for the cost vector
            X_mk_sel = X[units_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y[units_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))
            # Y_mk_sel_trueIdx_indices = extractDigits(Y_mk_sel_trueIdx) # for tf.gather_nd, extract the prob Pr(a_{l}|c,x;\theta)
            # Y_mk_sel_trueIdx_indices = extractDigits_addID(Y_mk_sel_trueIdx)

            x_extends_Tensor = Insert_AllCosts_3D_Format(costArray_Insert_Set, X_mk_sel, dim_EachStep)

            CostSet_PostProbTensor = Compute_PosterProb_costSet_BuiltIn(x_extends_Tensor, model, Y_mk_sel_trueIdx)
            ID_argmax = tf.math.argmax(CostSet_PostProbTensor)
            Pred_costSet = costArray_Insert_Set[ID_argmax]

            Dict_sel_costSet[(mk, sel)] = Pred_costSet
            Result_Pred_CostSet = Result_Pred_CostSet + Pred_costSet

    assert len(Result_Pred_CostSet) == df_CostVect_ID.shape[0]

    return np.array(Result_Pred_CostSet), Dict_sel_costSet

def CostVector_Inference_withMultiRows_withDict_BuiltIn_sqlite(model, X, Y, df_CostVect_ID, cost_disType, loss3_all_name_train, timeStep):
    ##### Since we do not know the true prior distribution of cost vector for a seller, so we consider
    ##### Uniform type, namely, Pr(c1) Pr(c2)... Uniform & Pr(c1, c2, ...) also Uniform, Thus, the prior
    ##### in Equation (2) all could be eliminated...
    assert cost_disType == 'Uniform'
    # timeStep = 3
    dim_EachStep = int(X.shape[1] / timeStep)  # dim_EachStep: 8, and 3 is the timeStep, if timeStep is not 3, should modify it. BE CAREFUL!!!
    VisitedTuple = []
    Dict_sel_costSet = {}  # Firstly, get the costVec for each seller
    Result_Pred_CostSet = []
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
    sqlite3.register_converter("array", convert_array) ## remember this!!!

    conn_loss3_all = sqlite3.connect(loss3_all_name_train, detect_types = sqlite3.PARSE_DECLTYPES)
    cur_loss3_all = conn_loss3_all.cursor()

    for i in range(df_CostVect_ID.shape[0]):  # tqdm(range(df_CostVect_ID.shape[0]))

        mk = df_CostVect_ID.loc[i, 'MarketID_period']
        sel = df_CostVect_ID.loc[i, 'userid_profile']

        if (mk, sel) not in VisitedTuple:
            VisitedTuple.append((mk, sel))
            # df_CostVect_ID_mkSel is only used for computing the unit number
            df_CostVect_ID_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)
                                                      & (df_CostVect_ID['userid_profile'] == sel)]

            units_idxSet = df_CostVect_ID_mkSel.index.tolist()
            if len(units_idxSet) > 1:
                assert is_arithmetic(units_idxSet) == True  ##Ensure that rows from one seller is close

            # assert len(costVec_Space) == len(costArray_Insert_Set)
            # mkSel_name = '-'.join([str(int(mk)), str(int(sel))])
            # with open(Path_dict_loss3_trainData_all + mkSel_name + '.json', 'r') as f:
            #     costArray_Insert_Set = json.load(f)

            cur_loss3_all.execute("SELECT arr FROM loss3_all_train WHERE market =? AND seller = ?",(mk, sel))
            costArray_Insert_Array = cur_loss3_all.fetchone()[0]

            # make inference for the cost vector
            X_mk_sel = X[units_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y[units_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))
            # Y_mk_sel_trueIdx_indices = extractDigits(Y_mk_sel_trueIdx) # for tf.gather_nd, extract the prob Pr(a_{l}|c,x;\theta)
            # Y_mk_sel_trueIdx_indices = extractDigits_addID(Y_mk_sel_trueIdx)

            x_extends_Tensor = Insert_AllCosts_3D_Format_Array(costArray_Insert_Array, X_mk_sel, dim_EachStep)

            CostSet_PostProbTensor = Compute_PosterProb_costSet_BuiltIn(x_extends_Tensor, model, Y_mk_sel_trueIdx)

            ID_argmax = tf.math.argmax(CostSet_PostProbTensor)
            Pred_costSet = list(costArray_Insert_Array[ID_argmax,:])

            Dict_sel_costSet[(mk, sel)] = Pred_costSet
            Result_Pred_CostSet = Result_Pred_CostSet + Pred_costSet

    assert len(Result_Pred_CostSet) == df_CostVect_ID.shape[0]

    return np.array(Result_Pred_CostSet), Dict_sel_costSet

def CostVector_Inference_withMultiRows_withDict_BuiltIn_sqlite_batch(model, X, Y, df_CostVect_ID, cost_disType, loss3_all_path_train, timeStep, batch):
    ##### Since we do not know the true prior distribution of cost vector for a seller, so we consider
    ##### Uniform type, namely, Pr(c1) Pr(c2)... Uniform & Pr(c1, c2, ...) also Uniform, Thus, the prior
    ##### in Equation (2) all could be eliminated...
    assert cost_disType == 'Uniform'
    # timeStep = 3
    dim_EachStep = int(X.shape[1] / timeStep)  # dim_EachStep: 8, and 3 is the timeStep, if timeStep is not 3, should modify it. BE CAREFUL!!!
    VisitedTuple = []
    Dict_sel_costSet = {}  # Firstly, get the costVec for each seller
    Result_Pred_CostSet = []
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
    sqlite3.register_converter("array", convert_array) ## remember this!!!

    conn_loss3_all = sqlite3.connect(loss3_all_path_train + 'loss3_all_batch{}.db'.format(batch), detect_types = sqlite3.PARSE_DECLTYPES)
    cur_loss3_all = conn_loss3_all.cursor()

    for i in range(df_CostVect_ID.shape[0]):  # tqdm(range(df_CostVect_ID.shape[0]))

        mk = df_CostVect_ID.loc[i, 'MarketID_period']
        sel = df_CostVect_ID.loc[i, 'userid_profile']

        if (mk, sel) not in VisitedTuple:
            VisitedTuple.append((mk, sel))
            # df_CostVect_ID_mkSel is only used for computing the unit number
            df_CostVect_ID_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)
                                                      & (df_CostVect_ID['userid_profile'] == sel)]

            units_idxSet = df_CostVect_ID_mkSel.index.tolist()
            if len(units_idxSet) > 1:
                assert is_arithmetic(units_idxSet) == True  ##Ensure that rows from one seller is close

            cur_loss3_all.execute("SELECT arr FROM loss3_all_train_batch_{} WHERE market =? AND seller = ?".format(batch),(mk, sel))
            costArray_Insert_Array = cur_loss3_all.fetchone()[0]

            # make inference for the cost vector
            X_mk_sel = X[units_idxSet, :]  # shape: 4*24
            Y_mk_sel = Y[units_idxSet, :]
            Y_mk_sel_trueIdx = list(np.argmax(Y_mk_sel, axis=1))

            x_extends_Tensor = Insert_AllCosts_3D_Format_Array(costArray_Insert_Array, X_mk_sel, dim_EachStep)
            CostSet_PostProbTensor = Compute_PosterProb_costSet_BuiltIn(x_extends_Tensor, model, Y_mk_sel_trueIdx)
            ID_argmax = tf.math.argmax(CostSet_PostProbTensor)
            Pred_costSet = list(costArray_Insert_Array[ID_argmax,:])

            Dict_sel_costSet[(mk, sel)] = Pred_costSet
            Result_Pred_CostSet = Result_Pred_CostSet + Pred_costSet

    assert len(Result_Pred_CostSet) == df_CostVect_ID.shape[0]

    return np.array(Result_Pred_CostSet), Dict_sel_costSet


def Construct_Sampled_CostVect_Space(unit_num, UB_sets_costVec, dis_pars):
    gap = dis_pars['gap']
    cost_u_set = []
    for u in range(unit_num):
       # cost_u_UB = np.min([UB_sets_costVec[u], dis_pars['price_max']])
        cost_u_UB = np.min(UB_sets_costVec[u] + [dis_pars['price_max']])
        cost_u = [i for i in range(1, int(cost_u_UB) + 1, gap)]
        cost_u_set.append(cost_u)
    # cost_u_set = Compute_CostVect_Space(unit_num, UB_sets_costVec, dis_pars)
    ##### satisfy this condition c1<c2<c3....  #####
    if unit_num == 1:
        Sampled_costVec_Set = list(itertools.product(*cost_u_set))
    elif unit_num == 2: ###### since the maximum value of unit number is 3 in our case, so we have,
        Sampled_costVec_Set = list(filter(lambda e: e[0] < e[1] and e[1] - e[0] <= 60, itertools.product(*cost_u_set)))
    elif unit_num == 3:
        Sampled_costVec_Set = list(filter(lambda e: e[0] < e[1] and e[1]  < e[2] and e[1] - e[0] <= 60 and e[2] - e[1] <= 60, itertools.product(*cost_u_set)))       
    else:
        print(' unit_num {} , Do you filter out those rows in our data?'.format(unit_num))
        assert 1 == 0
    # Sampled_costVec_Set = list(result)
    
    return  Sampled_costVec_Set

def Sample_Costs_ForLoss1(X_Cost_UB, df_CostVect_ID, dis_pars):
    
    VisitedTuple = []
    Result_SampledCosts = []
    for i in range(df_CostVect_ID.shape[0]): #tqdm(range(df_CostVect_ID.shape[0]))         
        mk = df_CostVect_ID.loc[i, 'MarketID_period']
        sel = df_CostVect_ID.loc[i, 'userid_profile']       
        if (mk, sel) not in VisitedTuple:    
            VisitedTuple.append((mk, sel))    
            # df_CostVect_ID_mkSel is only used for computing the unit number 
            df_CostVect_ID_mkSel = df_CostVect_ID.loc[(df_CostVect_ID['MarketID_period'] == mk)&(df_CostVect_ID['userid_profile'] == sel)]       
            
            unit_num = len(np.unique(df_CostVect_ID_mkSel['unit']))       
            units_idxSet = df_CostVect_ID_mkSel.index.tolist()    
            if len(units_idxSet) > 1:
                assert is_arithmetic(units_idxSet) == True ##Ensure that rows from one seller is close               

            Cost_UB_mk_sel = X_Cost_UB[units_idxSet]
            
            UnitsID_Set = list(df_CostVect_ID_mkSel['unit'])
            IdxVec_Set = [UnitsID_Set.index(ele) for ele in list(np.unique(UnitsID_Set))] # return a = [1,1,2,2,2,3] => [0, 2, 5]
            UB_sets_costVec = Cost_UB_mk_sel[IdxVec_Set]
            assert unit_num == len(UB_sets_costVec)
            
            costVec_Sampled_Space = Construct_Sampled_CostVect_Space(unit_num, UB_sets_costVec, dis_pars)            
            Sampled_costVec = random.choice(costVec_Sampled_Space)
            
            ############## convert into format of insert array
            dict_unitNum = {i: UnitsID_Set.count(i) for i in UnitsID_Set}   
            Sampled_costVec_New = []
            for key in dict_unitNum.keys(): # 1,2,3
                Sampled_costVec_New = Sampled_costVec_New + [Sampled_costVec[int(key)-1]] * dict_unitNum[key]
            assert len(Sampled_costVec_New) == len(UnitsID_Set)

            Result_SampledCosts = Result_SampledCosts + Sampled_costVec_New

    return np.array(Result_SampledCosts)

def insert_Costs_into_batchX(CostSampled_BatchTensor, X_BatchTensor, timeStep): ### 2-D dimension
    # timeStep = 3
    dim_each = int(X_BatchTensor.get_shape()[1]/timeStep)

    # concatenate together finally
    CostSampled_BatchTensor_copy = copy.deepcopy(CostSampled_BatchTensor)
    # check_X_timeStep1_valid = tf.subtract(CostSampled_BatchTensor_copy, X_BatchTensor[:,0:1])
    validInsert_Flag = tf.math.reduce_all(tf.equal(X_BatchTensor[:, 0:dim_each], tf.zeros_like(X_BatchTensor[:, 0:dim_each]))) #justify if tensor a == tensor b
    if validInsert_Flag:
        CostSampled_BatchTensor_copy = tf.zeros_like(CostSampled_BatchTensor_copy)
    x_exBatch_Tensor_timeStep1 = tf.keras.layers.concatenate([CostSampled_BatchTensor_copy, X_BatchTensor[:, 0:dim_each]], axis = -1)
   
    CostSampled_BatchTensor_copy = copy.deepcopy(CostSampled_BatchTensor)
    # check_X_timeStep2_valid = tf.subtract(CostSampled_BatchTensor_copy, X_BatchTensor[:,dim_each : (dim_each+1)])
    validInsert_Flag = tf.math.reduce_all(tf.equal(X_BatchTensor[:, dim_each:int(2*dim_each)], tf.zeros_like(X_BatchTensor[:, dim_each:int(2*dim_each)]))) #justify if tensor a == tensor b
    if validInsert_Flag:
        CostSampled_BatchTensor_copy = tf.zeros_like(CostSampled_BatchTensor_copy)
    x_exBatch_Tensor_timeStep2 = tf.keras.layers.concatenate([CostSampled_BatchTensor_copy, X_BatchTensor[:, dim_each:int(2*dim_each)]], axis = -1)
       
    CostSampled_BatchTensor_copy = copy.deepcopy(CostSampled_BatchTensor)
    # check_X_timeStep3_valid = tf.subtract(CostSampled_BatchTensor_copy, X_BatchTensor[:,2*dim_each : (2*dim_each+1)])
    validInsert_Flag = tf.math.reduce_all(tf.equal(X_BatchTensor[:, int(2*dim_each):int(3*dim_each)], tf.zeros_like(X_BatchTensor[:,int(2*dim_each):int(3*dim_each)]))) #justify if tensor a == tensor b
    if validInsert_Flag:
        CostSampled_BatchTensor_copy = tf.zeros_like(CostSampled_BatchTensor_copy)
    x_exBatch_Tensor_timeStep3 = tf.keras.layers.concatenate([CostSampled_BatchTensor_copy, X_BatchTensor[:,int(2*dim_each):int(3*dim_each)]], axis = -1)
       
    x_exBatch_Tensor_timeStep_temp = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep1, x_exBatch_Tensor_timeStep2], axis = -1)
    x_exBatch_Tensor = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep_temp, x_exBatch_Tensor_timeStep3], axis = -1)      
   
    return x_exBatch_Tensor

def Correct_logZero_Case(ProbArr):
    Result  = np.where(ProbArr > 1.0e-10, ProbArr, 1.0e-10)
    # Result = np.where(ProbArr > 1.0e-10, numpy.log10(ProbArr_tmp), -10)
    return  Result

def Insert_AllCosts_3D_Format(costArray_InsertSet, X_mk_sel, dim_EachStep):
    
    costArray_Insert_Tensor = tf.convert_to_tensor(costArray_InsertSet, dtype=float) #59*3, 3 is the row num
    costArray_Insert_TensorT = tf.transpose(costArray_Insert_Tensor) # 3 * 59
    # costArray_Insert_TensorT = tf.reshape(tf.expand_dims(costArray_Insert_TensorT, 0), [costArray_Insert_TensorT.get_shape()[0], -1, 1])
    costArray_Insert_TensorT = tf.expand_dims(costArray_Insert_TensorT, -1)
    # <tf.Tensor: shape=(3,), dtype=int32, numpy=array([3, 59,  1], dtype=int32)>
    x_mk_sel_tensor = tf.convert_to_tensor(X_mk_sel, dtype=float)
    x_extends = tf.tile(x_mk_sel_tensor, tf.constant([1, len(costArray_InsertSet)])) 
    # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 3, 1416], dtype=int32)>
    x_extends = tf.reshape(x_extends, [x_mk_sel_tensor.get_shape()[0], len(costArray_InsertSet), -1])
    # <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 3, 59, 24], dtype=int32)>

    costArray_Insert_TensorT_copy = copy.deepcopy(costArray_Insert_TensorT[:, :, 0:1])# Index not include the right value               
    validInsert_Flag = tf.math.reduce_all(tf.equal(x_extends[:, :, 0:dim_EachStep], tf.zeros_like(x_extends[:, :, 0:dim_EachStep]))) #justify if tensor a == tensor b
    if validInsert_Flag:
        costArray_Insert_TensorT_copy = tf.zeros_like(costArray_Insert_TensorT_copy)
    x_exBatch_Tensor_timeStep1 = tf.keras.layers.concatenate([costArray_Insert_TensorT_copy[:, :, 0:1], x_extends[:, :, 0:dim_EachStep]], axis=-1)

    costArray_Insert_TensorT_copy = copy.deepcopy(costArray_Insert_TensorT[:, :, 0:1])              
    validInsert_Flag = tf.math.reduce_all(tf.equal(x_extends[:, :, dim_EachStep:(2*dim_EachStep)], tf.zeros_like(x_extends[:, :, dim_EachStep:(2*dim_EachStep)]))) #justify if tensor a == tensor b
    if validInsert_Flag:
        costArray_Insert_TensorT_copy = tf.zeros_like(costArray_Insert_TensorT_copy)
    x_exBatch_Tensor_timeStep2 = tf.keras.layers.concatenate([costArray_Insert_TensorT_copy, x_extends[:, :, dim_EachStep:(2*dim_EachStep)]], axis=-1)

    costArray_Insert_TensorT_copy = copy.deepcopy(costArray_Insert_TensorT[:, :, 0:1])        
    validInsert_Flag = tf.math.reduce_all(tf.equal(x_extends[:, :, (2*dim_EachStep):(3*dim_EachStep)], tf.zeros_like(x_extends[:, :,(2*dim_EachStep):(3*dim_EachStep)]))) #justify if tensor a == tensor b
    if validInsert_Flag:
        costArray_Insert_TensorT_copy = tf.zeros_like(costArray_Insert_TensorT_copy)
    x_exBatch_Tensor_timeStep3 = tf.keras.layers.concatenate([costArray_Insert_TensorT_copy, x_extends[:, :, (2*dim_EachStep):(3*dim_EachStep)]], axis=-1)

    # concatenate together finally
    x_exBatch_Tensor_timeStep_temp = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep1, x_exBatch_Tensor_timeStep2], axis=-1)
    x_extends_Tensor = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep_temp, x_exBatch_Tensor_timeStep3], axis=-1)

    return x_extends_Tensor


def Insert_AllCosts_3D_Format_Array(costArray_InsertArray, X_mk_sel, dim_EachStep):
    costArray_Insert_Tensor = tf.convert_to_tensor(costArray_InsertArray, dtype=float)  # 59*3, 3 is the row num
    costArray_Insert_TensorT = tf.transpose(costArray_Insert_Tensor)  # 3 * 59
    # costArray_Insert_TensorT = tf.reshape(tf.expand_dims(costArray_Insert_TensorT, 0), [costArray_Insert_TensorT.get_shape()[0], -1, 1])
    costArray_Insert_TensorT = tf.expand_dims(costArray_Insert_TensorT, -1)
    # <tf.Tensor: shape=(3,), dtype=int32, numpy=array([3, 59,  1], dtype=int32)>
    x_mk_sel_tensor = tf.convert_to_tensor(X_mk_sel, dtype=float)
    x_extends = tf.tile(x_mk_sel_tensor, tf.constant([1, costArray_InsertArray.shape[0]]))
    # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 3, 1416], dtype=int32)>
    x_extends = tf.reshape(x_extends, [x_mk_sel_tensor.get_shape()[0], costArray_InsertArray.shape[0], -1])
    # <tf.Tensor: shape=(3,), dtype=int32, numpy=array([ 3, 59, 24], dtype=int32)>

    costArray_Insert_TensorT_copy = copy.deepcopy(
        costArray_Insert_TensorT[:, :, 0:1])  # Index not include the right value
    validInsert_Flag = tf.math.reduce_all(tf.equal(x_extends[:, :, 0:dim_EachStep], tf.zeros_like(
        x_extends[:, :, 0:dim_EachStep])))  # justify if tensor a == tensor b
    if validInsert_Flag:
        costArray_Insert_TensorT_copy = tf.zeros_like(costArray_Insert_TensorT_copy)
    x_exBatch_Tensor_timeStep1 = tf.keras.layers.concatenate(
        [costArray_Insert_TensorT_copy[:, :, 0:1], x_extends[:, :, 0:dim_EachStep]], axis=-1)

    costArray_Insert_TensorT_copy = copy.deepcopy(costArray_Insert_TensorT[:, :, 0:1])
    validInsert_Flag = tf.math.reduce_all(tf.equal(x_extends[:, :, dim_EachStep:(2 * dim_EachStep)], tf.zeros_like(
        x_extends[:, :, dim_EachStep:(2 * dim_EachStep)])))  # justify if tensor a == tensor b
    if validInsert_Flag:
        costArray_Insert_TensorT_copy = tf.zeros_like(costArray_Insert_TensorT_copy)
    x_exBatch_Tensor_timeStep2 = tf.keras.layers.concatenate(
        [costArray_Insert_TensorT_copy, x_extends[:, :, dim_EachStep:(2 * dim_EachStep)]], axis=-1)

    costArray_Insert_TensorT_copy = copy.deepcopy(costArray_Insert_TensorT[:, :, 0:1])
    validInsert_Flag = tf.math.reduce_all(tf.equal(x_extends[:, :, (2 * dim_EachStep):(3 * dim_EachStep)],
                                                   tf.zeros_like(x_extends[:, :, (2 * dim_EachStep):(
                                                               3 * dim_EachStep)])))  # justify if tensor a == tensor b
    if validInsert_Flag:
        costArray_Insert_TensorT_copy = tf.zeros_like(costArray_Insert_TensorT_copy)
    x_exBatch_Tensor_timeStep3 = tf.keras.layers.concatenate(
        [costArray_Insert_TensorT_copy, x_extends[:, :, (2 * dim_EachStep):(3 * dim_EachStep)]], axis=-1)

    # concatenate together finally
    x_exBatch_Tensor_timeStep_temp = tf.keras.layers.concatenate(
        [x_exBatch_Tensor_timeStep1, x_exBatch_Tensor_timeStep2], axis=-1)
    x_extends_Tensor = tf.keras.layers.concatenate([x_exBatch_Tensor_timeStep_temp, x_exBatch_Tensor_timeStep3],
                                                   axis=-1)

    return x_extends_Tensor


def Compute_CostVect_Space_Loss3(unit_num, Cost_UB_mkSel_eachUnit, dis_pars): 
    
    gap = dis_pars['gap']
    cost_u_set = []
    for u in range(unit_num):
        cost_u_UB = np.min([Cost_UB_mkSel_eachUnit[u], dis_pars['price_max']])
        cost_u = [i for i in range(1, int(cost_u_UB) + 1, gap)]
        cost_u_set.append(cost_u)
    # cost_u_set = Compute_CostVect_Space(unit_num, Cost_UB_mkSel_eachUnit, dis_pars)
    # costVec_space = list(itertools.product(*cost_u_set))

    if unit_num == 1:
        costVec_space = list(itertools.product(*cost_u_set))
    elif unit_num == 2: ###### since the maximum value of unit number is 3 in our case, so we have,
        # problem = problem.addConstraint(lambda a, b: a<b, ('c1', 'c2'))
        costVec_space = list(filter(lambda e: e[1] - e[0] <= 60, itertools.product(*cost_u_set)))
    elif unit_num == 3:
        costVec_space = list(filter(lambda e: e[1] - e[0] <= 60 and e[2] - e[1] <= 60, itertools.product(*cost_u_set)))       
    else:
        print('unit_num {} , Do you filter out those rows in our data?'.format(unit_num))
        assert 1 == 0   
    # costVec_space = list(resultAll)
    
    ##### satisfy this condition c1<c2<c3....  #####
    if unit_num == 1:
        costVec_space_cons = list(itertools.product(*cost_u_set))
    elif unit_num == 2: ###### since the maximum value of unit number is 3 in our case, so we have,
        costVec_space_cons = list(filter(lambda e: e[0] < e[1] and e[1] - e[0] <= 60, itertools.product(*cost_u_set)))
    elif unit_num == 3:
        costVec_space_cons = list(filter(lambda e: e[0] < e[1] and e[1] < e[2] and e[1] - e[0] <= 60 and e[2] - e[1] <= 60, itertools.product(*cost_u_set)))
    else:
        print('unit_num {} , Do you filter out those rows in our data?'.format(unit_num))
        assert 1 == 0
    
    return  costVec_space, costVec_space_cons

def Compute_Loss3_Component(x_extends_TensorAll, Y_mk_sel_trueIdx, model):
    Y_result_TensorAll = None
    for r in range(x_extends_TensorAll.get_shape()[0]):
        Yr_logits_Tensor = model(tf.reshape(x_extends_TensorAll[r, :, :], [x_extends_TensorAll.get_shape()[1], 3, -1]))
        Yr_probTensor_org = tf.gather(Yr_logits_Tensor, Y_mk_sel_trueIdx[r], axis = 1)
        Yr_probTensor = tf.clip_by_value(Yr_probTensor_org, 1e-10, 1)
        
        Yr_probTensor = tf.reshape(Yr_probTensor, [-1, 1])
        assert Yr_probTensor.get_shape()[0] == x_extends_TensorAll.get_shape()[1]
        if r == 0:
            Y_result_TensorAll = Yr_probTensor
        else:
            Y_result_TensorAll = tf.concat([Y_result_TensorAll, Yr_probTensor], axis = 1)
     
    Y_result_TensorAll_log = tf.math.log(Y_result_TensorAll)
    comp_temp = tf.math.reduce_sum(Y_result_TensorAll_log, axis = 1) #axis=1: add the elements from all columns
    comp_temp = tf.reshape(comp_temp, [comp_temp.get_shape()[0], 1])
    
    Efun_c_Tensor = tf.math.exp(comp_temp)
    Efun_c_Tensor = tf.clip_by_value(Efun_c_Tensor, 1e-10, 1)
     
    # Efun_c_Arr = tf.math.exp(comp_temp).numpy() #
    Efun_c_Arr = Efun_c_Tensor.numpy()
    ResultTensor = tf.reduce_sum(comp_temp * Efun_c_Arr, axis = 0)/np.sum(Efun_c_Arr)
     
    return ResultTensor, Efun_c_Arr

def compute_rows_costs_Yprobs(x_extends_TensorAll, model, Y_mk_sel_trueIdx):
    x_extends_Tensor_2d = tf.reshape(x_extends_TensorAll, [-1, x_extends_TensorAll.get_shape()[-1]])#3d->2d
    x_extends_Tensor_2d_inputModel = tf.reshape(x_extends_Tensor_2d, [x_extends_Tensor_2d.get_shape()[0], 3, -1])
    Y_logits_Tensor_org = model(x_extends_Tensor_2d_inputModel)
    Y_logits_Tensor = tf.clip_by_value(Y_logits_Tensor_org, 1e-10, 1)

    trueIdx_slices = list(np.repeat(Y_mk_sel_trueIdx, x_extends_TensorAll.get_shape()[1]))
    Y_result_Tensor_Temp = Tensor2d_slice(Y_logits_Tensor, trueIdx_slices)
    Y_result_TensorAll_trans = tf.reshape(Y_result_Tensor_Temp, [x_extends_TensorAll.get_shape()[0], -1])
    Y_result_TensorAll = tf.transpose(Y_result_TensorAll_trans)

    return Y_result_TensorAll


def Compute_Loss3_Component_BuiltIn(x_extends_TensorAll, Y_mk_sel_trueIdx, model):
    Y_result_TensorAll = compute_rows_costs_Yprobs(x_extends_TensorAll, model, Y_mk_sel_trueIdx)
    Y_result_TensorAll_log = tf.math.log(Y_result_TensorAll)
    comp_temp = tf.math.reduce_sum(Y_result_TensorAll_log, axis=1)  #axis=1: add the elements from all columns
    comp_temp = tf.reshape(comp_temp, [comp_temp.get_shape()[0], 1])

    Efun_c_Tensor = tf.math.exp(comp_temp)
    Efun_c_Tensor = tf.clip_by_value(Efun_c_Tensor, 1e-10, 1)

    # Efun_c_Arr = tf.math.exp(comp_temp).numpy()
    Efun_c_Arr = Efun_c_Tensor.numpy()
    ResultTensor = tf.reduce_sum(comp_temp * Efun_c_Arr, axis=0) / np.sum(Efun_c_Arr)

    return ResultTensor, Efun_c_Arr
    
    

    
    

            
            