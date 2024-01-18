#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:51:47 2023

@author: lvye
"""

import pandas as pd
import random, math
from utils import Compute_unique_mkSel_Pair

def divide_train_test_data(df, ratio):
             
    market_seller_set_unique = Compute_unique_mkSel_Pair(df)
    
    train_df = pd.DataFrame(columns = df.columns.tolist())
    valid_df = pd.DataFrame(columns = df.columns.tolist())
    test_df = pd.DataFrame(columns = df.columns.tolist())
    
    num = len(market_seller_set_unique)
    idx_set = list(range(num))
    random.shuffle(idx_set) 
    
    train_pair_idx = idx_set[0 : math.ceil(num * ratio)]
    valid_pair_idx = idx_set[math.ceil(num * ratio) : math.ceil(num * (0.5 * ratio + 0.5))]
    test_pair_idx = idx_set[math.ceil(num * (0.5 * ratio + 0.5)) : len(idx_set)]
    
    sel_item_pair_unique = list(market_seller_set_unique)
    train_pair, valid_pair, test_pair = [], [], []
    for i in range(len(sel_item_pair_unique)):
        if i in train_pair_idx:
            train_pair.append(sel_item_pair_unique[i])
        elif i in test_pair_idx:
            test_pair.append(sel_item_pair_unique[i])
        elif i in valid_pair_idx:
            valid_pair.append(sel_item_pair_unique[i])
        else:
            print('sth wrong in divide_train_test_data module !')
            assert(i == 0)   
    
    for mp, sel in train_pair:
        data_sel = df.loc[(df['MarketID_period'] == mp)&(df['userid_profile'] == sel)]
        train_df = pd.concat([train_df, data_sel], ignore_index = True)
        # train_df = train_df.append(data_sel, ignore_index = True)
    for mp, sel in valid_pair:
        data_sel = df.loc[(df['MarketID_period'] == mp)&(df['userid_profile'] == sel)]
        valid_df = pd.concat([valid_df, data_sel], ignore_index = True)
        # valid_df = valid_df.append(data_sel, ignore_index = True)
    for mp, sel in test_pair:
        data_sel = df.loc[(df['MarketID_period'] == mp)&(df['userid_profile'] == sel)]
        test_df = pd.concat([test_df, data_sel], ignore_index = True)
        # test_df = test_df.append(data_sel, ignore_index = True)
    assert((train_df.shape[0] + valid_df.shape[0] + test_df.shape[0]) == df.shape[0])
    
    return train_df, valid_df, test_df




file_name = 'Filtered_data_CDA_trans_IR'
file_path = './data/'
# df = pd.read_csv(file_path + file_name + '.csv', header = 0)
df_upper = pd.read_csv(file_path + file_name + '_UB.csv', header = 0)

##### First Compute Upper Bound for the data #####
# df = df.fillna(0)
# df_upper = Compute_upper_bound_unit_cost(df)
# df_upper.to_csv(file_path + file_name + '_UB.csv', index = False)

ratio = 0.8
for idx in range(1, 6):
    print('\n')
    print(idx)
    
    df_upper_train,df_upper_valid, df_upper_test = divide_train_test_data(df_upper, ratio)
    
    print(df_upper_train.shape)
    print(df_upper_valid.shape)
    print(df_upper_test.shape)
    
    df_upper_train.to_csv(file_path +'/Train'+str(idx)+'_' + file_name + '_UB.csv', index = False)
    df_upper_valid.to_csv(file_path +'/Valid'+str(idx)+'_' + file_name + '_UB.csv', index = False)
    df_upper_test.to_csv(file_path +'/Test' +str(idx)+'_'+ file_name + '_UB.csv', index = False)

    assert((df_upper_train.shape[0] + df_upper_valid.shape[0] + df_upper_test.shape[0]) == df_upper.shape[0])