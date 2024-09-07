#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:06:28 2024

@author: qb
"""

import pickle
import pandas as pd
import numpy as np 
import xgboost
import copy

class predict:
    def __init__(self, x_):
        self.x_ = x_
        self.reg = pickle.load(open('requires/xgb1_old', 'rb'))
        self.reg1 = pickle.load(open('requires/xgb1', 'rb'))
        self.regid = pickle.load(open('requires/lm_old.pkl', 'rb'))
        self.oe = pickle.load(open('requires/old_oe.pkl','rb'))
        
        
    def predict(self):
        #simple prediction
        print(self.x_.columns ,'****************************************************')
        cols = self.reg1.get_booster().feature_names
        self.x_=self.x_[cols]
        
        pr1=self.reg1.predict(self.x_)
        #pr2=self.reg.predict(self.x_)
        
        return pr1
        
    def predict_blended_lm(self, sigma, scaler, pred_, clean_data, x_d, td ,old_model = False, lm=False):
        'pred_ from xgb'
        #cols = self.regid.feature_names_in_
        #x_prime = copy.deepcopy(self.x_[cols])
        'if data provided by the user does not exists then lm will not perfom as we have no '
        'mechanism of out of sample missing value handling right now'
        'Here we are simply imputing the average of missing value as it is considered MRA (Missing not at random)'
        'we  can try to fill in the mean value'
        #idx = x_prime.isna().index 
        #x_prime = x_prime.drop(idx)
        if old_model:
            re = clean_data(x_d, x_pr = td, encoding ='one_hot')
            drop_columns = ['engine','seats','owner',
                    'index', 'fuel', 'transmission', 'km_driven', 'seller_type', 'company', 'avg_sellertype']
            re.drop(columns = drop_columns, inplace=True)
            re=re[self.reg.get_booster().feature_names]
            pre = self.reg.predict(re)
         
            if lm == True:
                fe = ['mileage', 'max_power' , 'co_cc', '_num_sales', '_seats_in_year']
                re[['year']]=self.oe.transform(re[['year']].values.reshape(-1,1))
                re[fe]=scaler.transform(re[fe].values.reshape([-1,5]))
                pred = self.regid.predict(re)
                print('going to return',pred)
                #regid gives us multiple predictions right now for blending we will tak only one
                return pre,pred

        #if x_prime.shape[0] > 0:
        #training the older xgboost model require features
    
        # else:
        #     r_=scaler.fit_transform([[pred]]).flatten()
        # else:
        #     return pred_
        
        ###define a new blend and validate it 
        #new curr model
        #due to shortage of time no could not make it work
    #     def predict_poly_ml(self, sigma, scaler, pred_):
    #         #scale the model appropretly and scale the model as well
    #         #both the scaler are the same for now
    #         if sigma == 0:
    #             return pred_
    #         else:
    #             curr_blend=self.curr * sigma + pred_
    #         return curr_blend
        
    # def pertubed_predict(self, sample_size, num_rep):
        #create quantiles here
        #random sample from test data =
        
        
        '''pertube perdiction'''
        '''sample_idx = []
        for i in range(sample_size):
            sample_idx.append(round(np.random.uniform(self.test.shape[0])))

        predictions =[]
        #mix data 
        for j in range(num_rep):
            for i in range(self.x_.shape[0]):
                idx = round(np.random.uniform(len(sample_idx)))
                test_ = self.test.loc[sample_idx]
                test_.loc[idx] = self.x_.loc[i]
                test_['year']=test_['year'].astype('int')
                test_['mileage'] = test_['mileage'].astype('float')
                pred=predictions.append(self.reg.predict(test_))
                predictions.append((pred[idx], 1))
        
        return predictions'''
    