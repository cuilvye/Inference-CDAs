#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:08:15 2023

@author: lvye
"""

# import sqlite3
import numpy as np
import json, os, time
import math, shelve
from tqdm import tqdm

from utils import Compute_unique_mkSelUnit_Pair
from utils import Compute_unique_mkSel_Pair
from inferences import Compute_CostVect_Space_Loss3
from inferences import Construct_insertCosts_Format

def CMF_Fitted_ContinuousC(sigma, c, a):  # i.e., Lemma 3
    cumu_prob = None  # if c<= 0
    if c > a:
        cumu_prob = 1
    elif c > 0 and c <= a:
        cumu_prob = math.exp(-(a / c - 1) / sigma)
    elif c <= 0:
        cumu_prob = 0

    return cumu_prob


def Compute_prob_cost_given_ask(ask, sigma, cost_set):
    ProbArray_cost_given_ask = np.zeros((len(cost_set), 2))
    for i in range(len(cost_set)):
        cost = cost_set[i]  # cost takes values from 1,2,3,..., 300, so we have cost - 1 as follows
        prob_cost = CMF_Fitted_ContinuousC(sigma, cost, ask) - CMF_Fitted_ContinuousC(sigma, cost - 1, ask)
        ProbArray_cost_given_ask[i, 0] = cost
        ProbArray_cost_given_ask[i, 1] = prob_cost

    return ProbArray_cost_given_ask


def Compute_unit_cost_distrib(u_asks_set, sigma, dis_pars):
    ### the space of unit cost variable ### need to rethink it !!
    dis_min, dis_max, gap = dis_pars['price_min'], dis_pars['price_max'], dis_pars['gap']
    cost_set = [i for i in range(dis_min, dis_max + 1, gap)]

    ProbArray_Cost_ask_Set = []
    for ask in u_asks_set:
        ProbArray_c_ask = Compute_prob_cost_given_ask(ask, sigma, cost_set)
        ProbArray_Cost_ask_Set.append(ProbArray_c_ask)

    # Lemma 4
    # What is the prior distribution of C^{jk}? its range?
    ProbArray_Cost_Prior = np.zeros((len(cost_set), 2))
    for i in range(len(cost_set)):
        cost = cost_set[i]
        ProbArray_Cost_Prior[i, 0] = cost
        ProbArray_Cost_Prior[i, 1] = 1 / len(cost_set)

    # Equation (10) in Lemma 4
    ResultArray = np.zeros((len(cost_set), 2))
    ResultArray[:, 0] = ProbArray_Cost_Prior[:, 0]

    result_nume = np.ones((len(cost_set), 1))
    for j in range(len(ProbArray_Cost_ask_Set)):
        prob_aj = ProbArray_Cost_ask_Set[j][:, 1].reshape(len(ProbArray_Cost_ask_Set[j][:, 1]), 1)
        result_nume = np.multiply(prob_aj, result_nume)  # np.multiply: element-wise multiply
    # (300,) is different from (300, 1), the previous one will be error when use multiply, so need to reshape into (, 1)    
    result_denum = np.power(ProbArray_Cost_Prior[:, 1].reshape(len(ProbArray_Cost_Prior[:, 1]), 1),
                            len(ProbArray_Cost_ask_Set) - 1)
    result_vec = np.divide(result_nume, result_denum)

    R_asks = np.sum(result_vec)
    Prob_c_given_asks = np.divide(result_vec, R_asks)
    ResultArray[:, 1] = Prob_c_given_asks.reshape((Prob_c_given_asks.shape[0],))

    return ResultArray  # first col is the C value, the second col is the computed probs


def Compute_Prior_distribution_unit_cost(df, sigma, dis_pars):
    priorDis_dict = {}

    market_seller_set = []
    for i in range(df.shape[0]):
        market_id = df.loc[i, 'MarketID_period']
        seller_id = df.loc[i, 'userid_profile']
        market_seller_set.append((market_id, seller_id))

    market_seller_set_unique = list(set(market_seller_set))
    print('I am computing the prior distribution of unit cost given sigma, please be patient...')
    # for mk, sel in market_seller_set_unique:   
    for i in tqdm(range(len(market_seller_set_unique))):
        # print(mk, sel)
        mk_sel_pair = market_seller_set_unique[i]
        mk, sel = mk_sel_pair[0], mk_sel_pair[1]

        df_mk_seller = df.loc[(df['MarketID_period'] == mk) & (df['userid_profile'] == sel)]
        df_mk_seller = df_mk_seller.reset_index(drop=True)

        unit_number = len(np.unique(df_mk_seller['unit']))
        for u in range(unit_number):
            unit_id = u + 1  # 'unit' col starts from 1 not 0!

            df_mk_seller_unit = df_mk_seller.loc[(df_mk_seller['unit']) == unit_id]

            dict_key = (mk, sel, unit_id)
            u_asks_set = list(df_mk_seller_unit['ask'])

            unit_cost_distribArray = Compute_unit_cost_distrib(u_asks_set, sigma, dis_pars)
            dict_val = unit_cost_distribArray
            priorDis_dict[dict_key] = dict_val

    return priorDis_dict


def Compute_Prior_distribution_unit_cost_v1(df, sigma, dis_pars):
    priorDis_dict = {}
    market_seller_set = []
    for i in range(df.shape[0]):
        market_id = df.loc[i, 'MarketID_period']
        seller_id = df.loc[i, 'userid_profile']
        market_seller_set.append((market_id, seller_id))

    market_seller_set_unique = list(set(market_seller_set))
    print('I am computing the prior distribution of unit cost given sigma, please be patient...')
    # for mk, sel in market_seller_set_unique:   
    for i in tqdm(range(len(market_seller_set_unique))):
        # print(mk, sel)
        mk_sel_pair = market_seller_set_unique[i]
        mk, sel = mk_sel_pair[0], mk_sel_pair[1]

        df_mk_seller = df.loc[(df['MarketID_period'] == mk) & (df['userid_profile'] == sel)]
        df_mk_seller = df_mk_seller.reset_index(drop=True)

        unit_number = len(np.unique(df_mk_seller['unit']))
        for u in range(unit_number):
            unit_id = u + 1  # 'unit' col starts from 1 not 0!

            df_mk_seller_unit = df_mk_seller.loc[(df_mk_seller['unit']) == unit_id]

            dict_key = (mk, sel, unit_id)
            u_asks_set = list(df_mk_seller_unit['ask'])

            unit_cost_distribArray = Compute_unit_cost_distrib(u_asks_set, sigma, dis_pars)
            dict_val = unit_cost_distribArray
            priorDis_dict[str(dict_key)] = dict_val # the difference from the previous version...!!!

    return priorDis_dict


def Compute_Prior_distribution_unit_cost_v2(df, sigma, dis_pars, save_name):
    # priorDis_dict = {}
    # with shelve.open(save_name, 'n') as priorDis_dict:
    #    print('I am creating a new db format...')

    market_seller_set = []
    for i in range(df.shape[0]):
        market_id = df.loc[i, 'MarketID_period']
        seller_id = df.loc[i, 'userid_profile']
        market_seller_set.append((market_id, seller_id))

    market_seller_set_unique = list(set(market_seller_set))
    print('I am computing the prior distribution of unit cost given sigma, please be patient...')
    # priorDis_dict =  shelve.open(save_name, 'w')
    with shelve.open(save_name, 'n') as priorDis_dict:  # save_name
        for i in tqdm(range(len(market_seller_set_unique))):
            # print(mk, sel)
            mk_sel_pair = market_seller_set_unique[i]
            mk, sel = mk_sel_pair[0], mk_sel_pair[1]

            df_mk_seller = df.loc[(df['MarketID_period'] == mk) & (df['userid_profile'] == sel)]
            df_mk_seller = df_mk_seller.reset_index(drop=True)

            unit_number = len(np.unique(df_mk_seller['unit']))
            for u in range(unit_number):
                unit_id = u + 1  # 'unit' col starts from 1 not 0!

                df_mk_seller_unit = df_mk_seller.loc[(df_mk_seller['unit']) == unit_id]

                dict_key = (mk, sel, unit_id)
                u_asks_set = list(df_mk_seller_unit['ask'])

                unit_cost_distribArray = Compute_unit_cost_distrib(u_asks_set, sigma, dis_pars)
                dict_val = unit_cost_distribArray
                priorDis_dict[str(dict_key)] = dict_val


# return priorDis_dict

def Compute_loss2_dependencies(df_train_new, dis_pars, priorDis_dict):
    dict_loss2_trainData = {}
    dict_loss2_trainData_priorDisArr = {}
    mkSelUnit_uniqueSet = Compute_unique_mkSelUnit_Pair(df_train_new)
    for i in tqdm(range(len(mkSelUnit_uniqueSet))):
        mk_sel_unit_pair = mkSelUnit_uniqueSet[i]
        mk, sel, unit = mk_sel_unit_pair

        df_mkSelUnit = df_train_new.loc[(df_train_new['MarketID_period'] == mk)
                                        & (df_train_new['userid_profile'] == sel)
                                        & (df_train_new['unit'] == unit)]

        Cost_UB_mkSelUnit_Set = df_mkSelUnit['upper_bound_unit_cost']
        Cost_UB_mkSelUnit_Unique = list(np.unique(Cost_UB_mkSelUnit_Set))
        assert len(Cost_UB_mkSelUnit_Unique) == 1
        Cost_UB_mkSelUnit = Cost_UB_mkSelUnit_Unique[0]

        cost_u_UB = np.min([Cost_UB_mkSelUnit, dis_pars['price_max']])
        cost_u = [i for i in range(1, int(cost_u_UB) + 1, dis_pars['gap'])]
        cost_uInsert_Set = []  ## difference <= 60  will affect this??? Be careful!!!
        for cost in cost_u:
            costsInsert = [cost] * df_mkSelUnit.shape[0]
            cost_uInsert_Set.append(costsInsert)
        dict_loss2_trainData[str((mk, sel, unit))] = cost_uInsert_Set

        ### since dump array -> list ###
        priorDis_mkSelUnit = np.array(priorDis_dict[str((mk, sel, unit))])
        priorDis_mkSelUnit_realSet = []
        for cost in cost_u:
            cost_u_idx = np.where(priorDis_mkSelUnit[:, 0] == cost)[0].tolist()[0]
            priorDis_mkSelUnit_realSet.append(priorDis_mkSelUnit[cost_u_idx, 1])
        priorDis_mkSelUnit_realArr = np.array(priorDis_mkSelUnit_realSet).reshape(len(priorDis_mkSelUnit_realSet), 1)

        dict_loss2_trainData_priorDisArr[str((mk, sel, unit))] = priorDis_mkSelUnit_realArr

    return dict_loss2_trainData, dict_loss2_trainData_priorDisArr

def load_or_compute_loss2_dependencies(file_dict_loss2_Data, file_dict_loss2_Data_priorDisArr, df, dis_pars, prior_name, sigma):
    if os.path.exists(file_dict_loss2_Data):
        print('I am reloading the costSpace for loss 2, please be patient......')
        dict_loss2_Data = read_dict_file(file_dict_loss2_Data)
        dict_loss2_Data_priorDisArr = read_dict_file(file_dict_loss2_Data_priorDisArr)
    else:
        priorDis_dict = load_or_compute_priorSigma(prior_name, df, sigma, dis_pars)
        print('I am preparing the costSpace for loss 2, please be patient......')
        dict_loss2_Data, dict_loss2_Data_priorDisArr = Compute_loss2_dependencies(df, dis_pars, priorDis_dict)
        save_dict_file(file_dict_loss2_Data, dict_loss2_Data)
        save_dict_file(file_dict_loss2_Data_priorDisArr, dict_loss2_Data_priorDisArr)

    return dict_loss2_Data, dict_loss2_Data_priorDisArr

def load_or_compute_loss2_dependencies_sqlite(file_dict_loss2_Data, file_dict_loss2_Data_priorDisArr, df, dis_pars, prior_name, sigma):
    if os.path.exists(file_dict_loss2_Data):
        print('I am reloading the costSpace for loss 2, please be patient......')
        dict_loss2_Data = read_dict_file(file_dict_loss2_Data)
        dict_loss2_Data_priorDisArr = read_dict_file(file_dict_loss2_Data_priorDisArr)
    else:
        priorDis_dict = load_or_compute_priorSigma(prior_name, df, sigma, dis_pars)
        print('I am preparing the costSpace for loss 2, please be patient......')
        dict_loss2_Data, dict_loss2_Data_priorDisArr = Compute_loss2_dependencies(df, dis_pars, priorDis_dict)
        save_dict_file(file_dict_loss2_Data, dict_loss2_Data)
        save_dict_file(file_dict_loss2_Data_priorDisArr, dict_loss2_Data_priorDisArr)

    return dict_loss2_Data, dict_loss2_Data_priorDisArr

def Compute_Save_loss3_dependencies(df_train_new, Path_dict_loss3_trainData_all, Path_dict_loss3_trainData_cons, dis_pars):
    mkSel_uniqueSet = Compute_unique_mkSel_Pair(df_train_new)
    for i in tqdm(range(len(mkSel_uniqueSet))):
        mk_sel_pair = mkSel_uniqueSet[i]
        mk, sel = mk_sel_pair[0], mk_sel_pair[1]
        # for mk, sel in mkSel_uniqueSet:
        df_mkSel = df_train_new.loc[(df_train_new['MarketID_period'] == mk)
                                    & (df_train_new['userid_profile'] == sel)]
        Costs_UB_mkSel = np.array(df_train_new['upper_bound_unit_cost'])

        UnitsID_Set = list(df_mkSel['unit'])
        IdxVec_Set = [UnitsID_Set.index(ele) for ele in set(UnitsID_Set)]  # return a = [1,1,2,2,2,3] => [0, 2, 5]
        Cost_UB_mkSel_eachUnit = Costs_UB_mkSel[IdxVec_Set]

        unit_num = len(np.unique(UnitsID_Set))
        assert unit_num == len(Cost_UB_mkSel_eachUnit)

        costVec_Space, costVec_Space_Cons = Compute_CostVect_Space_Loss3(unit_num, Cost_UB_mkSel_eachUnit, dis_pars)
        UnitsID_Array = np.array(UnitsID_Set)
        costArray_InsertSet = Construct_insertCosts_Format(costVec_Space, UnitsID_Array)
        costArray_InsertSet_Cons = Construct_insertCosts_Format(costVec_Space_Cons, UnitsID_Array)

        save_name = '-'.join([str(int(mk)), str(int(sel))])
        with open(Path_dict_loss3_trainData_all + save_name + '.json', "w") as f:
            json.dump(costArray_InsertSet, f)

        with open(Path_dict_loss3_trainData_cons + save_name + '.json', "w") as f:
            json.dump(costArray_InsertSet_Cons, f)

def prepare_for_loss3(Path_dict_loss3_Data_all, Path_dict_loss3_Data_cons, df, dis_pars):
    if not os.path.exists(Path_dict_loss3_Data_all):
        os.makedirs(Path_dict_loss3_Data_all)
    if not os.path.exists(Path_dict_loss3_Data_cons):
        os.makedirs(Path_dict_loss3_Data_cons)
    time_s = time.time()
    if len(os.listdir(Path_dict_loss3_Data_all)) == 0:
        print('I am preparing the costSpace for loss 3, please be patient......')
        Compute_Save_loss3_dependencies(df, Path_dict_loss3_Data_all, Path_dict_loss3_Data_cons, dis_pars)
        time_e = time.time()
        elapsed_l = (time_e - time_s) / 60.0
        print("this prepare takes epoch took {:.4} minutes".format(elapsed_l), end="\n")

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def save_dict_file(filename, Data):
    with open(filename, 'w') as outfile:
        json.dump(Data, outfile, default=default)


def read_dict_file(filename):
    with open(filename) as infile:
        data = json.load(infile)
    return data

def load_or_compute_priorSigma(prior_name, df, sigma, dis_pars):
    if os.path.exists(prior_name):
        priorDis_dict = read_dict_file(prior_name)
    else:
        priorDis_dict = Compute_Prior_distribution_unit_cost_v1(df, sigma, dis_pars)
        save_dict_file(prior_name, priorDis_dict)

    return priorDis_dict


# def default(obj):
#     if type(obj).__module__ == np.__name__:
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return obj.item()
#     raise TypeError('Unknown type:', type(obj))
#
#
# # dumped = json.dumps(data, default=default)
#
# def save_dict_file(filename, Data):
#     with open(filename, 'w') as outfile:
#         json.dump(Data, outfile, default=default)
#
#
# def read_dict_file(filename):
#     with open(filename) as infile:
#         data = json.load(infile)
#     return data
#
# def load_or_compute_priorSigma(prior_name, df, sigma, dis_pars):
#     if os.path.exists(prior_name):
#         priorDis_dict = read_dict_file(prior_name)
#     else:
#         priorDis_dict = Compute_Prior_distribution_unit_cost_v1(df, sigma, dis_pars)
#         save_dict_file(prior_name, priorDis_dict)
#
#     return priorDis_dict
#
# def load_or_compute_loss2_dependencies(file_dict_loss2_Data, file_dict_loss2_Data_priorDisArr, df, dis_pars, prior_name, sigma):
#     if os.path.exists(file_dict_loss2_Data):
#         print('I am reloading the costSpace for loss 2, please be patient......')
#         dict_loss2_Data = read_dict_file(file_dict_loss2_Data)
#         dict_loss2_Data_priorDisArr = read_dict_file(file_dict_loss2_Data_priorDisArr)
#     else:
#         priorDis_dict = load_or_compute_priorSigma(prior_name, df, sigma, dis_pars)
#         print('I am preparing the costSpace for loss 2, please be patient......')
#         dict_loss2_Data, dict_loss2_Data_priorDisArr = Compute_loss2_dependencies(df, dis_pars, priorDis_dict)
#         save_dict_file(file_dict_loss2_Data, dict_loss2_Data)
#         save_dict_file(file_dict_loss2_Data_priorDisArr, dict_loss2_Data_priorDisArr)
#
#     return dict_loss2_Data, dict_loss2_Data_priorDisArr
#
# def prepare_for_loss3(Path_dict_loss3_Data_all, Path_dict_loss3_Data_cons, df, dis_pars):
#     if not os.path.exists(Path_dict_loss3_Data_all):
#         os.makedirs(Path_dict_loss3_Data_all)
#     if not os.path.exists(Path_dict_loss3_Data_cons):
#         os.makedirs(Path_dict_loss3_Data_cons)
#     time_s = time.time()
#     if len(os.listdir(Path_dict_loss3_Data_all)) == 0:
#         print('I am preparing the costSpace for loss 3, please be patient......')
#         Compute_Save_loss3_dependencies(df, Path_dict_loss3_Data_all, Path_dict_loss3_Data_cons, dis_pars)
#         time_e = time.time()
#         elapsed_l = (time_e - time_s) / 60.0
#         print("this prepare takes epoch took {:.4} minutes".format(elapsed_l), end="\n")