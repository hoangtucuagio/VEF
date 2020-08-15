import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
import math
import xgboost as xgb
# from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score

import joblib


list_A = ['60172--21060'
,'60127--21060'
,'60126--21050'
,'60058--21050'
,'60052--21060']

def file_selector(folder_path=r'.\input_test'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Chọn File', filenames)
    return os.path.join(folder_path, selected_filename)

st.title("MÔ HÌNH DỰ ĐOÁN TOI TƯƠNG LAI CHO KHÁCH HÀNG CÁ NHÂN TRONG LĨNH VỰC NGÂN HÀNG")
filename = file_selector()
# st.write('You selected `%s`' % filename)

# @st.cache
def load_dp_model():
    xgr_toidp = open("./xgr_toidp_group01.pkl","rb")
    xgr_toidp_scaler = open("./xgr_toidp_group01_scaler_rb.pkl","rb")
    model_load = joblib.load(xgr_toidp) 
    scaler_rb_load = joblib.load(xgr_toidp_scaler)
    return model_load,scaler_rb_load

# @st.cache
def load_loan_model():
    xgr_toiloan = open("./xgb_toiloan_group01.pkl","rb")
    xgr_toiloan_scaler = open("./xgb_toiloan_group01_scaler.pkl","rb")	
    model_load = joblib.load(xgr_toiloan) 
    scaler_rb_load = joblib.load(xgr_toiloan_scaler)	
    return model_load,scaler_rb_load

# @st.cache
def proc_deposit(scaler_model):	
    df = pd.read_csv(filename)
    df.drop(df[df['AMT_CUR']<=1].index,inplace=True)
    PRODUCT_CDE_df_temp = pd.get_dummies(df['PRODUCT_CDE'], prefix = 'PRODUCT')
    PRODUCT_CDE_df_temp.insert(loc=9,column='PRODUCT_10100--1003',value=0)
    PRODUCT_CDE_df_temp.insert(loc=12,column='PRODUCT_11026--1003',value=0)
    df['TOI_LOG'] = pd.to_numeric(np.log(df['TOI']), errors='coerce')
    df['AMT_CUR_LOG'] = pd.to_numeric(np.log(df['AMT_CUR']), errors='coerce')
    temp2 = pd.concat([PRODUCT_CDE_df_temp,df], axis = 1)
    temp2 = temp2.reset_index(drop=True)
    b = PRODUCT_CDE_df_temp.columns.tolist()
    numerical_columns_test = b + ['AMT_CUR_LOG','ACCT_USE_DAYS','INTEREST_RATE','RATE_FTP']
    temp2.columns = ['PRODUCT_-1--1003', 'PRODUCT_10011--1003', 'PRODUCT_10013--1003',
       'PRODUCT_10015--1003', 'PRODUCT_10020--1003', 'PRODUCT_10032--1003',
       'PRODUCT_10073--1003', 'PRODUCT_10075--1003', 'PRODUCT_10079--1003',
       'PRODUCT_10100--1003', 'PRODUCT_11011--1003', 'PRODUCT_11015--1003',
       'PRODUCT_11026--1003', 'PRODUCT_11032--1003', 'CUSTOMER_CDE',
       'PRODUCT_CDE', 'ACCT_ID', 'AMT_INIT', 'AMT_CUR', 'INTEREST_RATE',
       'RATE_FTP', 'LMV', 'LBV', 'ACCT_USE_DAYS', 'TOI', 'PROCESS_MONTH',
       'PROCESS_YEAR', 'TOI_LOG', 'AMT_CUR_LOG']
    s_rb_test=scaler_model.transform(temp2[numerical_columns_test])
    df_mm_test = pd.DataFrame(s_rb_test, columns=numerical_columns_test)
    y_test = temp2['TOI']
    return df,df_mm_test,y_test

# @st.cache
def proc_loan(scaler_model):
    df_test = pd.read_csv(filename)	
    df_b_test = df_test[(df_test.LOAITRAGOP =='E') | (df_test.LOAITRAGOP =='B')]
    df_b_test = df_b_test.reset_index(drop=True)
    df_b_test.drop(df_b_test[df_b_test['AMT_CUR']<=1].index,inplace=True)
    df_b_test = df_b_test[df_b_test['PRODUCT_CDE'].isin(list_A)]
    df_b_test = df_b_test.reset_index(drop=True)
    df_b_test = df_b_test[(df_b_test['TOI']>=1000) & (df_b_test['TOI']<=20000000)]
    df_b_test = df_b_test.reset_index(drop=True)
    df_b_test['TOI_LOG'] = pd.to_numeric(np.log(df_b_test['TOI']), errors='coerce')
    df_b_test['AMT_CUR_LOG'] = pd.to_numeric(np.log(df_b_test['AMT_CUR']), errors='coerce')
    PRODUCT_CDE_df_test = pd.get_dummies(df_b_test['PRODUCT_CDE'], prefix = 'PRODUCT_CDE')
    PRODUCT_CDE_df_test.insert(loc=0,column='PRODUCT_CDE_60036--21060',value=0)	
    # df_b_test.drop(['PRODUCT_CDE'],axis=1, inplace = True)
    col_test = ['AMT_CUR_LOG', 'ACCT_USE_DAYS', 'RATE_FTP', 'INTEREST_RATE','KYHAN']
    df_b_test_02 = pd.concat([PRODUCT_CDE_df_test,df_b_test[col_test]], axis = 1)
    s_rb2=scaler_model.transform(df_b_test_02)
    df_mm_test = pd.DataFrame(s_rb2, columns=df_b_test_02.columns)
    y_test = df_b_test['TOI']	
    return df_b_test,df_mm_test,y_test

# @st.cache
def get_data():
    if("toidp" in filename):
        model,scaler_model = load_dp_model()
        df,X_test,y_test = proc_deposit(scaler_model)
    if("toiloan" in filename):
        model,scaler_model = load_loan_model()
        df,X_test,y_test = proc_loan(scaler_model)
    return df,X_test,y_test,model

df,X_test,y_test,model= get_data()
y_pred_rf2 = model.predict(X_test)
y_pred_rf2 = np.exp(y_pred_rf2)
st.write(f'Tổng sai lệch (RMSE)           = {mean_squared_error(y_test, y_pred_rf2)**0.5:.4f}')
st.write(f'Độ phù hợp của mô hình (R_SQUARED)  = {r2_score(y_test, y_pred_rf2)*100:.4f} %')

y_test = y_test.reset_index(drop=True)
y_test2_ser = pd.Series(y_test)
y_pred_rf2_ser = pd.Series(y_pred_rf2)
data_result3  = pd.concat([y_pred_rf2_ser, y_test2_ser], axis=1, keys=['TOI_PREDICT', 'TOI_ACTUAL'])

data_result3['TOI_STD']=abs(data_result3['TOI_ACTUAL']-data_result3['TOI_PREDICT'])

data_result3.sort_values(by='TOI_STD',ascending=False).head(20)

result_df = df.copy()
result_df.drop(['TOI_LOG','AMT_CUR_LOG'],axis=1, inplace = True)
result_df['TOI_PREDICT'] = y_pred_rf2
result_df['TOI_6M_NEXT'] = (result_df['TOI_PREDICT']/result_df['ACCT_USE_DAYS'])*180
result_df['TOI_ONEYEAR_NEXT'] = (result_df['TOI_PREDICT']/result_df['ACCT_USE_DAYS'])*365
# df.insert(loc=0,column='TOI_PREDICT',value=y_pred_rf2)

# def read_file():    
#     return pd.read_csv(filename) 

# df = read_file()

min_year = int(result_df['PROCESS_MONTH'].min())
max_year = int(result_df['PROCESS_MONTH'].max())

prod = result_df['PRODUCT_CDE'].unique()

'## Kết quả dự đoán TOI theo sản phẩm'
product = st.selectbox('Sản phẩm', prod)
result_df[result_df['PRODUCT_CDE'] == product]


'## Kết quả dự đoán TOI theo tháng (năm 2020)'
year = st.slider('Tháng', min_year, max_year)
result_df[result_df['PROCESS_MONTH'] == year]

if st.button('Lưu File'):
    open('ketqua_toi.csv', 'w').write(result_df.to_csv())