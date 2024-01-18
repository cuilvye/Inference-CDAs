#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random, math
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

def Compute_unique_mkSel_Pair(df):
    market_seller_set = []
    for i in range(df.shape[0]):
         market_id = df.loc[i, 'MarketID_period']
         seller_id = df.loc[i, 'userid_profile']
         market_seller_set.append((market_id, seller_id))
         
    market_seller_set_unique = list(set(market_seller_set))
    # a =[(1,2),(1,2),(3,4)] ==> b=list(np.unique(a)) [1, 2, 3, 4], Error, so use set()!
    # so np.unique(A) A=[1,2,3,3,4], not suitable for [(1,2),(1,2),(3,4)]
    
    return market_seller_set_unique

def Compute_unique_mkSelUnit_Pair(df):
    market_seller_unit_set = []
    for i in range(df.shape[0]):
         market_id = df.loc[i, 'MarketID_period']
         seller_id = df.loc[i, 'userid_profile']
         unit_id = df.loc[i, 'unit']
         market_seller_unit_set.append((market_id, seller_id, unit_id))
         
    market_seller_unit_set_unique = list(set(market_seller_unit_set))
    
    return market_seller_unit_set_unique

def Compute_upper_bound_unit_cost(df):
    # market_seller_set = []
    # for i in range(df.shape[0]):
    #      market_id = df.loc[i, 'MarketID_period']
    #      seller_id = df.loc[i, 'userid_profile']
    #      market_seller_set.append((market_id, seller_id))
         
    # market_seller_set_unique = list(set(market_seller_set))
    market_seller_set_unique = Compute_unique_mkSel_Pair(df)

    df_upper = pd.DataFrame(columns = df.columns.values.tolist())
    upper_bound_set = []

    # for mk, sel in market_seller_set_unique:  
    print('I am computing the UB of private unit cost, please be patient...')
    for i in tqdm(range(len(market_seller_set_unique))):
        # print(mk, sel)
        mk_sel_pair = market_seller_set_unique[i]
        mk, sel = mk_sel_pair[0], mk_sel_pair[1]
        
        df_mk_seller = df.loc[(df['MarketID_period'] == mk)&(df['userid_profile'] == sel)]
        df_mk_seller = df_mk_seller.reset_index(drop=True)
        
        unit_number = len(np.unique(df_mk_seller['unit']))   
           
        for u in range(unit_number):
            df_mk_seller_unit = df_mk_seller.loc[(df_mk_seller['unit']) == u+1] # 'unit' col starts from 1 not 0!
            df_mk_seller_unit = df_mk_seller_unit.reset_index(drop=True)
            
            upper_bound_u = np.min(df_mk_seller_unit['ask'])
            for j in range(df_mk_seller_unit.shape[0]):  
                df_upper = pd.concat([df_upper, df_mk_seller_unit.loc[[j]]], ignore_index = True)
                upper_bound_set.append(upper_bound_u) 
    
    assert len(upper_bound_set) == df_upper.shape[0]   
    assert df_upper.shape[0] == df.shape[0]     
    df_upper['upper_bound_unit_cost'] = upper_bound_set
    
    return df_upper

def Determine_features_RNN(df):
    
    cols_name_set = df.columns.values.tolist()
    cols_name_set_sel = copy.deepcopy(cols_name_set)  
    cols_name_set_remove =[
                 'h3-cost', 'h2-cost', 'h1-cost',
                 'a2', 'a3', 'b2', 'b3', ## a1(the minimum of other sellers' asks) and b1(the largest bid value among buyers)
                 'h3-a2', 'h3-a3','h3-b2', 'h3-b3',
                 'h2-a2', 'h2-a3','h2-b2', 'h2-b3',
                 'h1-a2', 'h1-a3','h1-b2', 'h1-b3',
                 'h3-b1', 'h3-a1', 'h3-trans-mean', 'h3-trans-max','h3-trans-min','h3-trans-median', 'h3-trans-num', # timestep=3, for h3, lack h4,so remove all h3-contexts
                 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min', 'h2-trans-median', 'h2-trans-num',
                 'h1-trans-mean', 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num',
                 'trans-mean', 'trans-max', 'trans-min', 'trans-median','trans-num'
                 ]
    for name in cols_name_set_remove:
        cols_name_set_sel.remove(name)
        
    ## print(cols_name_set_sel)#len(cols_name_set_X) = 24, input_dim of each timestep is 8+1(cost), timestep = 3 
    #['h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min', 
    # 'h2-trans-median', 'h2-trans-num', 'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean', 
    # 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 'h1', 'b1',
    # 'a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num'] 
    ## print(cols_name_set_sel)#len(cols_name_set_X) = 9, input_dim of each timestep is 3+1(cost), timestep = 3 
    #['h3', 'h2-b1', 'h2-a1',  'h2', 'h1-b1', 'h1-a1',  'h1', 'b1','a1', ] 
    df_sel = df[cols_name_set_sel]
    
    cols_name_set_notX = ['MarketID_period', 'userid_profile', 'unit', 
                          'ask', 'cost', 'upper_bound_unit_cost']
    cols_name_set_X = copy.deepcopy(cols_name_set_sel)
    for name in cols_name_set_notX:
        cols_name_set_X.remove(name)
        
    assert (len(cols_name_set_X) + len(cols_name_set_notX)) == len(cols_name_set_sel)
    seq_dim = len(cols_name_set_X)   
    
    return df_sel, seq_dim


def Determine_features_RNN_moreFeas(df):
    cols_name_set = df.columns.values.tolist()
    cols_name_set_sel = copy.deepcopy(cols_name_set)
    cols_name_set_remove = [
        'h3-cost', 'h2-cost', 'h1-cost',
        'a2', 'a3', 'b2', 'b3',  ## a1(the minimum of other sellers' asks) and b1(the largest bid value among buyers)
        'h3-a2', 'h3-a3', 'h3-b2', 'h3-b3',
        'h2-a2', 'h2-a3', 'h2-b2', 'h2-b3',
        'h1-a2', 'h1-a3', 'h1-b2', 'h1-b3',
        'h3-b1', 'h3-a1', 'h3-trans-mean', 'h3-trans-max', 'h3-trans-min', 'h3-trans-median', 'h3-trans-num',
        # timestep=3, for h3, lack h4,so remove all h3-contexts
        'h2-trans-median', 'h2-trans-num',
        'h1-trans-median', 'h1-trans-num',
         'trans-median', 'trans-num' #'trans-mean', 'trans-max', 'trans-min',
    ]
    for name in cols_name_set_remove:
        cols_name_set_sel.remove(name)

    ## print(cols_name_set_sel)#len(cols_name_set_X) = 24, input_dim of each timestep is 8+1(cost), timestep = 3
    # ['h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min',
    # 'h2-trans-median', 'h2-trans-num', 'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean',
    # 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 'h1', 'b1',
    # 'a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']
    ## print(cols_name_set_sel)#len(cols_name_set_X) = 9, input_dim of each timestep is 3+1(cost), timestep = 3
    # ['h3', 'h2-b1', 'h2-a1',  'h2', 'h1-b1', 'h1-a1',  'h1', 'b1','a1', ]
    df_sel = df[cols_name_set_sel]

    cols_name_set_notX = ['MarketID_period', 'userid_profile', 'unit',
                          'ask', 'cost', 'upper_bound_unit_cost']
    cols_name_set_X = copy.deepcopy(cols_name_set_sel)
    for name in cols_name_set_notX:
        cols_name_set_X.remove(name)

    assert (len(cols_name_set_X) + len(cols_name_set_notX)) == len(cols_name_set_sel)
    seq_dim = len(cols_name_set_X)

    return df_sel, seq_dim

def extract_batchData_sameMarketSeller(df, bt_size): 
    
    # mk_sel_pair = []
    
    # mk_col_idx = list(df).index('MarketID_period') # list(dataframe): a list of columns name
    # slr_col_idx = list(df).index('userid_profile')
    # for i in range(df.shape[0]):
    #     mk = df.iloc[i, mk_col_idx]
    #     sel = df.iloc[i, slr_col_idx]
    #     mk_sel_pair.append((mk, sel))
    mk_sel_pair_unique = Compute_unique_mkSel_Pair(df)
      
    num = len(mk_sel_pair_unique)
    idx_set = list(range(num))
    random.shuffle(idx_set) 
      
    numUpdates = math.ceil(num / bt_size) # int: small one? should use math.ceil???
    
    batch_pairIdx, batch_dataFrame = [], []
    # mk_sel_pair_unique = list(mk_sel_pair_unique)
    print('I am dividing data into batches (extract_batchData_sameMarketSeller), please be patient...')
    for batch in range(0, numUpdates): # for batch in tqdm(range(0, numUpdates)):
        start  = batch * bt_size
        end  =  start + bt_size
        
        batch_i = idx_set[start:end]
        
        batch_i_mkSeller_pair = []
        for idx in batch_i:
            batch_i_mkSeller_pair.append(mk_sel_pair_unique[idx])
        
        batch_i_dataFrame = pd.DataFrame(columns = df.columns.tolist())
        for mk, sel in batch_i_mkSeller_pair:
            data_sel = df.loc[(df['MarketID_period'] == mk)&(df['userid_profile'] == sel)]
            batch_i_dataFrame = pd.concat([batch_i_dataFrame, data_sel], ignore_index = True)

        batch_pairIdx.append(batch_i_mkSeller_pair)
        batch_dataFrame.append(batch_i_dataFrame)
    
    check_num = 0
    for batch in batch_pairIdx:
        check_num = check_num + len(batch)
    
    assert(check_num == num)   
    
    return batch_dataFrame, batch_pairIdx


def extract_fixed_batchData_sameMarketSeller(df, bt_size):
    # mk_sel_pair = []

    # mk_col_idx = list(df).index('MarketID_period') # list(dataframe): a list of columns name
    # slr_col_idx = list(df).index('userid_profile')
    # for i in range(df.shape[0]):
    #     mk = df.iloc[i, mk_col_idx]
    #     sel = df.iloc[i, slr_col_idx]
    #     mk_sel_pair.append((mk, sel))
    mk_sel_pair_unique = Compute_unique_mkSel_Pair(df)

    num = len(mk_sel_pair_unique)
    idx_set = list(range(num))
    # random.shuffle(idx_set)

    numUpdates = math.ceil(num / bt_size)  # int: small one? should use math.ceil???

    batch_pairIdx, batch_dataFrame = [], []
    # mk_sel_pair_unique = list(mk_sel_pair_unique)
    print('I am dividing data into batches (extract_batchData_sameMarketSeller), please be patient...')
    for batch in range(0, numUpdates):  # for batch in tqdm(range(0, numUpdates)):
        start = batch * bt_size
        end = start + bt_size

        batch_i = idx_set[start:end]

        batch_i_mkSeller_pair = []
        for idx in batch_i:
            batch_i_mkSeller_pair.append(mk_sel_pair_unique[idx])

        batch_i_dataFrame = pd.DataFrame(columns=df.columns.tolist())
        for mk, sel in batch_i_mkSeller_pair:
            data_sel = df.loc[(df['MarketID_period'] == mk) & (df['userid_profile'] == sel)]
            batch_i_dataFrame = pd.concat([batch_i_dataFrame, data_sel], ignore_index=True)

        batch_pairIdx.append(batch_i_mkSeller_pair)
        batch_dataFrame.append(batch_i_dataFrame)

    check_num = 0
    for batch in batch_pairIdx:
        check_num = check_num + len(batch)

    assert (check_num == num)

    return batch_dataFrame, batch_pairIdx

## this need to be careful and think more about it  ##
def Convert_Y_into_categorical(batch_idataFrame, i):     
     y_org = batch_idataFrame.loc[i, 'ask']
     b_max = batch_idataFrame.loc[i, 'b1']
     a_min = batch_idataFrame.loc[i, 'a1']
     
     y_label = None
     #should check b_max first, if it not satisfy then go to a_min !!
     if y_org <= b_max and b_max > 0: 
         y_label = 0     # 0: accept current largest bid    
     elif (a_min > 0 and y_org >= a_min) or (a_min == 0 and y_org > b_max and b_max > 0):
             y_label = 1 # 1: decline current largest bid
     elif y_org > b_max and y_org < a_min:
        ratio = (y_org - b_max)/(a_min - b_max)
        if ratio > 0 and ratio <= 0.1:
            y_label = 2
        elif ratio > 0.1 and ratio <= 0.2:
            y_label = 3
        elif ratio > 0.2 and ratio <= 0.3:
           y_label = 4
        elif ratio > 0.3 and ratio <= 0.4:
           y_label = 5
        elif ratio > 0.4 and ratio <= 0.5:
           y_label = 6
        elif ratio > 0.5 and ratio <= 0.6:
           y_label = 7
        elif ratio > 0.6 and ratio <= 0.7:
            y_label = 8
        elif ratio > 0.7 and ratio <= 0.8:
           y_label = 9
        elif ratio > 0.8 and ratio <= 0.9:
           y_label = 10
        elif ratio > 0.9 and ratio < 1.0:
           y_label = 11
     elif  b_max == 0 and a_min == 0:
         if y_org > 0  and y_org <= 100:
             y_label = 12
         elif y_org > 100 and y_org <= 200:
            y_label = 13
         elif y_org > 200 and y_org <=300:
            y_label = 14
         elif y_org > 300:
            y_label = 15
                   
     return y_label


def Convert_Y_into_categorical_LessClass(batch_idataFrame, i):
    y_org = batch_idataFrame.loc[i, 'ask']
    b_max = batch_idataFrame.loc[i, 'b1']
    a_min = batch_idataFrame.loc[i, 'a1']

    y_label = None
    # should check b_max first, if it not satisfy then go to a_min !!
    if y_org <= b_max and b_max > 0:
        y_label = 0  # 0: accept current largest bid
    elif (a_min > 0 and y_org >= a_min) or (a_min == 0 and y_org > b_max and b_max > 0):
        y_label = 1  # 1: decline current largest bid
    elif y_org > b_max and y_org < a_min:
        ratio = (y_org - b_max) / (a_min - b_max)
        if ratio > 0 and ratio <= 0.2:
            y_label = 2
        elif ratio > 0.2 and ratio <= 0.4:
            y_label = 3
        elif ratio > 0.4 and ratio <= 0.6:
            y_label = 4
        elif ratio > 0.6 and ratio <= 0.8:
            y_label = 5
        elif ratio > 0.8 and ratio <= 1.0:
            y_label = 6
    elif b_max == 0 and a_min == 0:
        if y_org > 0 and y_org <= 100:
            y_label = 7
        elif y_org > 100 and y_org <= 200:
            y_label = 8
        elif y_org > 200 and y_org <= 300:
            y_label = 9
        elif y_org > 300:
            y_label = 10

    return y_label

def Convert_Y_into_categorical_LessClass2(batch_idataFrame, i):
    y_org = batch_idataFrame.loc[i, 'ask']
    b_max = batch_idataFrame.loc[i, 'b1']
    a_min = batch_idataFrame.loc[i, 'a1']

    y_label = None
    # should check b_max first, if it not satisfy then go to a_min !!
    if y_org <= b_max and b_max > 0:
        y_label = 0  # 0: accept current largest bid
    elif (a_min > 0 and y_org >= a_min) or (a_min == 0 and y_org > b_max and b_max > 0):
        y_label = 1  # 1: decline current largest bid
    elif y_org > b_max and y_org < a_min:
        ratio = (y_org - b_max) / (a_min - b_max)
        if ratio > 0 and ratio <= 0.3:
            y_label = 2
        elif ratio > 0.3 and ratio <= 0.6:
            y_label = 3
        elif ratio > 0.6 and ratio <= 0.9:
            y_label = 4
        elif ratio > 0.9 and ratio < 1.0:
            y_label = 5
    elif b_max == 0 and a_min == 0:
        if y_org > 0 and y_org <= 150:
            y_label = 6
        elif y_org > 150 and y_org <= 300:
            y_label = 7
        elif y_org > 300:
            y_label = 8
    return y_label

def Convert_Y_into_categorical_LessClass3(batch_idataFrame, i):
    y_org = batch_idataFrame.loc[i, 'ask']
    b_max = batch_idataFrame.loc[i, 'b1']
    a_min = batch_idataFrame.loc[i, 'a1']

    y_label = None
    # should check b_max first, if it not satisfy then go to a_min !!
    if y_org <= b_max and b_max > 0:
        y_label = 0  # 0: accept current largest bid
    elif (a_min > 0 and y_org >= a_min) or (a_min == 0 and y_org > b_max and b_max > 0):
        y_label = 1  # 1: decline current largest bid
    elif y_org > b_max and y_org < a_min:
        ratio = (y_org - b_max) / (a_min - b_max)
        if ratio > 0 and ratio <= 0.5:
            y_label = 2
        elif ratio > 0.5 and ratio <= 0.75:
            y_label = 3
        elif ratio > 0.75 and ratio < 1.0:
            y_label = 4
    elif b_max == 0 and a_min == 0:
        # if y_org > 0 and y_org <= 150:
        #     y_label = 5
        # elif y_org > 150 and y_org <= 300:
        #     y_label = 6
        # elif y_org > 300:
            y_label = 5
    return y_label

def Compute_ask_interval(y_org, a_min, b_max):
    ratio = 0
    if a_min > 0 and y_org >= a_min:
        ratio = (y_org - a_min)/a_min
    elif a_min == 0 and y_org > b_max and b_max > 0:
        ratio = (y_org - b_max)/b_max
    return ratio

def Convert_Y_into_categorical_LessClass4(batch_idataFrame, i):
     y_org = batch_idataFrame.loc[i, 'ask']
     b_max = batch_idataFrame.loc[i, 'b1']
     a_min = batch_idataFrame.loc[i, 'a1']

     y_label = None
     # should check b_max first, if it not satisfy then go to a_min !!
     if y_org <= b_max and b_max > 0:
         y_label = 0  # 0: accept current largest bid
     elif (a_min > 0 and y_org >= a_min) or (a_min == 0 and y_org > b_max and b_max > 0):
         ratio = Compute_ask_interval(y_org, a_min, b_max)
         if ratio >= 0 and ratio <= 0.1:
             y_label = 1
         elif ratio > 0.1 and ratio <= 0.4:
             y_label = 2
         elif ratio > 0.4:
              y_label = 3       
     elif y_org > b_max and y_org < a_min:
         ratio = (y_org - b_max) / (a_min - b_max)
         if ratio > 0 and ratio <= 0.5:
             y_label = 4
         elif ratio > 0.5 and ratio <= 0.75:
             y_label = 5
         elif ratio > 0.75 and ratio < 1.0:
             y_label = 6
     elif b_max == 0 and a_min == 0:
         # if y_org > 0 and y_org <= 150:
         #     y_label = 5
         # elif y_org > 150 and y_org <= 300:
         #     y_label = 6
         # elif y_org > 300:
             y_label = 7
     return y_label    

def Convert_Y_into_categorical_Class6(batch_idataFrame, i, classes):
    y_org = batch_idataFrame.loc[i, 'ask']
    assert classes == 6
    assert y_org <= 300
    if y_org > 0 and y_org <= 50:
        y_label = 0
    elif y_org > 50 and y_org <= 100:
        y_label = 1
    elif y_org > 100 and y_org <= 150:
        y_label = 2
    elif y_org > 150 and y_org <= 200:
        y_label = 3
    elif y_org > 200 and y_org <= 250:
        y_label = 4
    else: #y_org > 250 and y_org <= 300:
        y_label = 5

    return y_label

def Convert_Y_into_categorical_Class7(batch_idataFrame, i, classes):
    y_org = batch_idataFrame.loc[i, 'ask']
    assert classes == 7
    # assert y_org <= 300
    if y_org > 0 and y_org <= 50:
        y_label = 0
    elif y_org > 50 and y_org <= 100:
        y_label = 1
    elif y_org > 100 and y_org <= 150:
        y_label = 2
    elif y_org > 150 and y_org <= 200:
        y_label = 3
    elif y_org > 200 and y_org <= 250:
        y_label = 4
    elif y_org > 250 and y_org <= 300:
        y_label = 5
    else:
        y_label = 6

    return y_label

def Convert_to_training_DataFormat(batch_idataFrame, classes):
    
    df_market_seller_unit = batch_idataFrame[['MarketID_period', 'userid_profile', 'unit']] #used for mark seller info   
    df_cost = batch_idataFrame[['cost']]     
    df_UB = batch_idataFrame[['upper_bound_unit_cost']]
    ############################## extract X ##############################
    cols_name_set_X = [
        'h3', 'h2-b1', 'h2-a1', 
        'h2', 'h1-b1', 'h1-a1', 
        'h1', 'b1', 'a1']
    # cols_name_set_X = [
    #     'h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min','h2-trans-median', 'h2-trans-num', 
    #     'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean','h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 
    #     'h1', 'b1','a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']        
    # print(cols_name_set_X)#len(cols_name_set_X) = 24, input_dim of each timestep is 8+1(cost), timestep = 3 
    #['h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min', 
    # 'h2-trans-median', 'h2-trans-num', 'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean', 
    # 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 'h1', 'b1',
    # 'a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']    
    df_X = batch_idataFrame[cols_name_set_X]
    
    ############################## extract y and convert it into categorical value ##############################    
    df_Y = batch_idataFrame[['ask']]
    Y_categorical = []   
    for i in range(batch_idataFrame.shape[0]):
        y_i = Convert_Y_into_categorical_LessClass4(batch_idataFrame, i)
        Y_categorical.append(y_i)
    
    assert (df_market_seller_unit.shape[1] + df_cost.shape[1]+ df_UB.shape[1] + df_X.shape[1] + df_Y.shape[1]) == batch_idataFrame.shape[1]
        
    return np.array(df_X), np.array(Y_categorical), np.array(df_cost), np.array(df_UB), df_market_seller_unit


def Convert_to_training_DataFormat_Class6(batch_idataFrame, classes):
    df_market_seller_unit = batch_idataFrame[
        ['MarketID_period', 'userid_profile', 'unit']]  # used for mark seller info
    df_cost = batch_idataFrame[['cost']]
    df_UB = batch_idataFrame[['upper_bound_unit_cost']]
    ############################## extract X ##############################
    cols_name_set_X = [
        'h3', 'h2-b1', 'h2-a1',
        'h2', 'h1-b1', 'h1-a1',
        'h1', 'b1', 'a1']
    # cols_name_set_X = [
    #     'h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min','h2-trans-median', 'h2-trans-num',
    #     'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean','h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num',
    #     'h1', 'b1','a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']
    # print(cols_name_set_X)#len(cols_name_set_X) = 24, input_dim of each timestep is 8+1(cost), timestep = 3
    # ['h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min',
    # 'h2-trans-median', 'h2-trans-num', 'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean',
    # 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 'h1', 'b1',
    # 'a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']
    df_X = batch_idataFrame[cols_name_set_X]

    ############################## extract y and convert it into categorical value ##############################
    df_Y = batch_idataFrame[['ask']]
    Y_categorical = []
    for i in range(batch_idataFrame.shape[0]):
        y_i = Convert_Y_into_categorical_Class6(batch_idataFrame, i, classes)
        Y_categorical.append(y_i)

    assert (df_market_seller_unit.shape[1] + df_cost.shape[1] + df_UB.shape[1] + df_X.shape[1] + df_Y.shape[1]) == batch_idataFrame.shape[1]

    return np.array(df_X), np.array(Y_categorical), np.array(df_cost), np.array(df_UB), df_market_seller_unit

def Convert_to_training_DataFormat_Class7(batch_idataFrame, classes):
    df_market_seller_unit = batch_idataFrame[
        ['MarketID_period', 'userid_profile', 'unit']]  # used for mark seller info
    df_cost = batch_idataFrame[['cost']]
    df_UB = batch_idataFrame[['upper_bound_unit_cost']]
    ############################## extract X ##############################
    cols_name_set_X = [
        'h3', 'h2-b1', 'h2-a1',
        'h2', 'h1-b1', 'h1-a1',
        'h1', 'b1', 'a1']
    # cols_name_set_X = [
    #     'h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min','h2-trans-median', 'h2-trans-num',
    #     'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean','h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num',
    #     'h1', 'b1','a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']
    # print(cols_name_set_X)#len(cols_name_set_X) = 24, input_dim of each timestep is 8+1(cost), timestep = 3
    # ['h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min',
    # 'h2-trans-median', 'h2-trans-num', 'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean',
    # 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 'h1', 'b1',
    # 'a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']
    df_X = batch_idataFrame[cols_name_set_X]

    ############################## extract y and convert it into categorical value ##############################
    df_Y = batch_idataFrame[['ask']]
    Y_categorical = []
    for i in range(batch_idataFrame.shape[0]):
        y_i = Convert_Y_into_categorical_Class7(batch_idataFrame, i, classes)
        Y_categorical.append(y_i)

    # print(df_market_seller_unit.shape[1])
    # print(df_cost.shape[1])
    # print(df_UB.shape[1])
    # print(df_X.shape[1])
    # print(df_Y.shape[1])
    # print(batch_idataFrame.shape[1])
    # print(list(batch_idataFrame))
    assert (df_market_seller_unit.shape[1] + df_cost.shape[1] + df_UB.shape[1] + df_X.shape[1] + df_Y.shape[1]) == batch_idataFrame.shape[1]

    return np.array(df_X), np.array(Y_categorical), np.array(df_cost), np.array(df_UB), df_market_seller_unit

def Convert_to_training_DataFormat_moreFeas(batch_idataFrame, classes):
    df_market_seller_unit = batch_idataFrame[
        ['MarketID_period', 'userid_profile', 'unit']]  # used for mark seller info
    df_cost = batch_idataFrame[['cost']]
    df_UB = batch_idataFrame[['upper_bound_unit_cost']]
    ############################## extract X ##############################
    cols_name_set_X = [
        'h3', 'h2-b1', 'h2-a1', 'h2-trans-max', 'h2-trans-min', 'h2-trans-mean',
        'h2', 'h1-b1', 'h1-a1','h1-trans-max', 'h1-trans-min', 'h1-trans-mean',
        'h1', 'b1', 'a1', 'h1-trans-max', 'h1-trans-min','h1-trans-mean']
    # cols_name_set_X = [
    #     'h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min','h2-trans-median', 'h2-trans-num',
    #     'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean','h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num',
    #     'h1', 'b1','a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']
    # print(cols_name_set_X)#len(cols_name_set_X) = 24, input_dim of each timestep is 8+1(cost), timestep = 3
    # ['h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min',
    # 'h2-trans-median', 'h2-trans-num', 'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean',
    # 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 'h1', 'b1',
    # 'a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']
    df_X = batch_idataFrame[cols_name_set_X]

    ############################## extract y and convert it into categorical value ##############################
    df_Y = batch_idataFrame[['ask']]
    Y_categorical = []
    for i in range(batch_idataFrame.shape[0]):
        y_i = Convert_Y_into_categorical_LessClass4(batch_idataFrame, i)
        Y_categorical.append(y_i)

    assert (df_market_seller_unit.shape[1] + df_cost.shape[1] + df_UB.shape[1] + df_X.shape[1] + df_Y.shape[1]) == \
           batch_idataFrame.shape[1]

    return np.array(df_X), np.array(Y_categorical), np.array(df_cost), np.array(df_UB), df_market_seller_unit

# def Convert_to_training_DataFormat(batch_idataFrame, classes):
    
#     df_market_seller_unit = batch_idataFrame[['MarketID_period', 'userid_profile', 'unit']] #used for mark seller info   
#     df_cost = batch_idataFrame[['cost']]     
#     ############################## extract X ##############################
#     cols_name_set = batch_idataFrame.columns.values.tolist()
#     cols_name_set_X = copy.deepcopy(cols_name_set)    
#     ## decide the X-features in the RNN model, e.g.,  
#     for name in ['MarketID_period', 'userid_profile', 'unit', 
#                  'cost', 
#                  'ask',
#                  'h3-cost', 'h2-cost', 'h1-cost',
#                  'a2', 'a3', 'b2', 'b3', ## a1(the minimum of other sellers' asks) and b1(the largest bid value among buyers)
#                  'h3-a2', 'h3-a3','h3-b2', 'h3-b3',
#                  'h2-a2', 'h2-a3','h2-b2', 'h2-b3',
#                  'h1-a2', 'h1-a3','h1-b2', 'h1-b3',
#                  'h3-b1', 'h3-a1', 'h3-trans-mean', 'h3-trans-max','h3-trans-min','h3-trans-median', 'h3-trans-num', # timestep=3, for h3, lack h4,so remove all h3-contexts
#                  ]:
#         cols_name_set_X.remove(name)
        
#     print(cols_name_set_X)#len(cols_name_set_X) = 24, input_dim of each timestep is 8+1(cost), timestep = 3 
#     #['h3', 'h2-b1', 'h2-a1', 'h2-trans-mean', 'h2-trans-max', 'h2-trans-min', 
#     # 'h2-trans-median', 'h2-trans-num', 'h2', 'h1-b1', 'h1-a1', 'h1-trans-mean', 
#     # 'h1-trans-max', 'h1-trans-min', 'h1-trans-median', 'h1-trans-num', 'h1', 'b1',
#     # 'a1', 'trans-mean', 'trans-max', 'trans-min', 'trans-median', 'trans-num']    
#     df_X = batch_idataFrame[cols_name_set_X]
    
#     ############################## extract y and convert it into categorical value ##############################    
#     df_Y = batch_idataFrame[['ask']]
#     Y_categorical = []
    
#     for i in range(batch_idataFrame.shape[0]):
#         y_i = Convert_Y_into_categorical(batch_idataFrame, i)
#         Y_categorical.append(y_i)
      
#     return np.array(df_X), np.array(Y_categorical), np.array(df_cost), df_market_seller_unit








