
import numpy as np
import pandas as pd
import joblib
from category_encoders import TargetEncoder

tg_encoder = joblib.load('target_encoder.pkl')
def preprocess_input(df):

  encoding_cols=['Provider','BeneID','AttendingPhysician','ClmDiagnosisCode_1']
  
  df.loc[:,encoding_cols]=tg_encoder.transform(df[encoding_cols])
  df['DOB']=pd.to_datetime(df['DOB']).dt.year
  df['age']=2009-df['DOB']
  df.drop('DOB',axis=1,inplace=True)
  df['ClaimStartDt']=pd.to_datetime(df['ClaimStartDt'])
  df['ClaimEndDt']=pd.to_datetime(df['ClaimEndDt'])
  df['clm_duration']=(df['ClaimEndDt']-df['ClaimStartDt']).dt.days
  df.drop(['ClaimStartDt','ClaimEndDt'],axis=1,inplace=True)
  df['AdmissionDt']=pd.to_datetime(df['AdmissionDt'],errors='coerce')
  df['DischargeDt']=pd.to_datetime(df['DischargeDt'],errors='coerce')
  
  df['LOS']=(df['DischargeDt']-df['AdmissionDt']).dt.days
  df['LOS'].fillna(0,inplace=True)
  df.drop(['AdmissionDt','DischargeDt'],axis=1,inplace=True)

  return df
