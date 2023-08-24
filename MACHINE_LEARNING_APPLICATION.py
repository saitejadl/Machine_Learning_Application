# """<?... MLAPP ...?>"""

# ------------------- Frame work and depency libraries -------------------
import streamlit as st
import json
from functools import lru_cache
import requests
from streamlit_lottie import st_lottie
import time

# ------------------- Manipulation of data -------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Imputation techniques -------------------
from sklearn.impute import SimpleImputer
# from sklearn.impute import IterativeImputer
# from sklearn.impute import KNNImputer

# ------------------- standradization techniques -------------------
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# ------------------- Encoding Techniques -------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# ------------------- Train Test split, Evaluation -------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,cohen_kappa_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# ---------------------------------------------------------------------------------------------------------------------------------------------

url = requests.get('https://assets4.lottiefiles.com/packages/lf20_1kHda4nbr8.json')
url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in URL")

#______________________________________________________________sidebar setup_____________________________________________________________

st.sidebar.header("𝕄𝔸ℂℍ𝕀ℕ𝔼 𝕃𝔼𝔸ℝℕ𝕀ℕ𝔾 𝔸ℙℙ𝕃𝕀ℂ𝔸𝕋𝕀𝕆ℕ")
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=True)

dictionary = {'Classification': ['Logistic', 'RandomForest', 'DecisionTree'], 'Regression': ['Linear','RandomForest', 'DecisionTree']}
# st.sidebar.title("𝕄𝔸ℂℍ𝕀ℕ𝔼 𝕃𝔼𝔸ℝℕ𝕀ℕ𝔾 𝔸ℙℙ𝕃𝕀ℂ𝔸𝕋𝕀𝕆ℕ")
selected_section = st.sidebar.selectbox("TYPE OF PROBLEM:", dictionary.keys())
selected_page = st.sidebar.selectbox("MODEL:", dictionary[selected_section])

if st.sidebar.button("Run model",use_container_width=True):
  st.sidebar.write(f"Running {selected_section}:{selected_page}")
  with st.sidebar:
    with st.spinner("Loading..."):
      st_lottie(url_json)
      time.sleep(2)
      st.sidebar.title("ⓂⒺⓉⓇⒾⒸⓈ")
      st.sidebar.success("Accuracy")
      st.sidebar.success("Recall")
      st.sidebar.success("Precission")
      st.sidebar.balloons()
      st.sidebar.snow()

#-----------------------------------------------------------------------------------------------------------------------------------------------
def stats(data):
  """
  Parameters  :   Dataframe
  Return      :   head(), tail(), dtype,  describe()
  in a single dataframe for convinient and compact representation
  """
  col_dtype=pd.DataFrame(data.dtypes,columns=['dtype'])
  col_desc=data.describe()
  return pd.concat([data.head(3),data.tail(3),col_dtype.T,col_desc])
#_____________________________________________________________________
def null_unique(data):
  """
  Parameters  :   Dataframe
  Return      :   nunique(),  isna()
  in a single dataframe for convinient and compact representation
  """
  uni=pd.DataFrame(data.nunique(),columns=['unique values'])
  uni['unique values %'] = (uni['unique values']/data.shape[0])*100
  nul=pd.DataFrame(data.isna().sum(),columns=['null values'])
  nul['null values %'] = (nul['null values']/data.shape[0])*100
  return pd.concat([uni,nul],axis=1)
#____________________________________________________________________
def dmy_hms(data,columns):
  """
  Parameters: datetime [ns] column
  Return    : day month year    hour minute second
  and appends these columns to the given dataframe
  """
  for column in columns:
    data[column+'_day'] = data[column].dt.date.apply(lambda x: x.strftime("%a"))
    data[column+'_month'] = data[column].dt.date.apply(lambda x: x.strftime("%b"))
    data[column+'_year'] = data[column].dt.year
    data[column+'_hour'] = data[column].dt.hour
    data[column+'_minute'] = data[column].dt.minute
    data[column+'_second'] = data[column].dt.second
    for i in [column+'_day',column+'_month',column+'_year',column+'_hour',column+'_minute',column+'_second']:
      if(data[i].nunique()==1):
        data.drop(i,inplace=True,axis=1)
    data.drop(column,axis=1,inplace=True)
#__________________________________________________________________
def type_cast(data,cat=None,num=None,da_ti=None):
  """
  Parameters: Dataframe, list of column name in dataframe to be converted to category , list of column name in dataframe to be converted to int64, list of column name in dataframe to be converted to datetime64[ns]
  Return    : Modified dataframe of given datatypes of corresponding columns list
  """
  if cat!=None:
    for i in cat:
      data[i] = data[i].astype('category')
  if num!=None:
    for j in num:
      data[j] = pd.to_numeric(data[j])
  if da_ti!=None:
    for k in da_ti:
      data[k] = pd.to_datetime(data[k])
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def clf_eval(model_name,y_act,y_pred):
  """
  Parameters:  model name : str, actual column, predicted column
  Return    :  accuracy, recall, precision, fi score, display confusion matric
  """
  print(model_name,"\n","_"*len(model_name)*3)
  print("accuracy: ",accuracy_score(y_act,y_pred))
  print("recall: ",recall_score(y_act,y_pred))
  print("precision: ",precision_score(y_act,y_pred))
  print("f1 score: ",f1_score(y_act,y_pred))
  # cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_act, y_pred), display_labels = [False, True])
  # cm_display.plot()
  # plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def reg_eval(model_name,y_train,train_y_pred,y_test,test_y_pred):
  """
  Parameters:  model name : str, train actual column, train predicted column, test actual column, test predicted column
  Return    :  MAE,RMSE of both tain and test , difference between train rmse and test rmse
  """
  print(model_name,"\n","_"*len(model_name)*3)
  print("train MAE: {} \n".format(mean_absolute_error(y_train,train_y_pred)))
  print("test MAE: {} \n".format(mean_absolute_error(y_test,test_y_pred)))
  print("-"*50)
  print("train RMSE: {} \n".format(np.sqrt(mean_squared_error(y_train,train_y_pred))))
  print("test RMSE: {} \n".format(np.sqrt(mean_squared_error(y_test,test_y_pred))))
  print("-"*50)
  print("difference in train and test rmse: {}".format(np.sqrt(mean_squared_error(y_train,train_y_pred)) - np.sqrt(mean_squared_error(y_test,test_y_pred))))

# st.set_page_config(page_title="MLAPP",page_icon="🧑‍💻", layout="centered"|"wide",initial_sideba_state='expanded)
# uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
train=pd.DataFrame()
test=pd.DataFrame()
subm=pd.DataFrame()

# @st.cache_resource()
def read_file(file_name,enc=None):
  df = pd.read_csv(uploaded_file, encoding=enc)
  return df

for uploaded_file in uploaded_files:
  if 'csv' in uploaded_file.type:
    if 'train' in uploaded_file.name.lower():
      st.write('Train Data')
      train = read_file(uploaded_file, enc='iso-8859-1')
      # train = pd.read_csv(uploaded_file)
      st.dataframe(train, use_container_width=True)
    elif 'test' in uploaded_file.name.lower():
      st.write('Test Data')
      test = read_file(uploaded_file)
      # test = pd.read_csv(uploaded_file)
      st.dataframe(test, use_container_width=True)
    elif 'subm' in uploaded_file.name.lower():
      st.write('submission Data')
      subm = read_file(uploaded_file)
      st.dataframe(subm, use_container_width=True)
    else:
      st.write(uploaded_file.name)
      st.write('Train Data')      
      train = read_file(uploaded_file)
      st.dataframe(train, use_container_width=True)

# @st.cache_resource()#suppress_st_warning=True)
def selection():
  cols = train.columns
  cat_opt = st.multiselect('What are your categorical columns',cols)
  num_opt = st.multiselect('What are your numerical columns',[x for x in cols if x not in cat_opt])
  date_opt = st.multiselect('What are your datetime columns',[x for x in cols if x not in num_opt+cat_opt])
  # NOTE if 'if st.button' is not there: output will be standby in webpage
  if st.button(label='Take', use_container_width=False):
    type_cast(train, cat = cat_opt, num = num_opt)
    st.dataframe(stats(train), use_container_width=True)
    st.dataframe(null_unique(train), use_container_width=True)
selection()
#______________________________________________________________search bar setup_____________________________________________________________
st.markdown('#')
search=st.text_input("",placeholder='FILE:query')
# @st.cache_resource()
def query(search):
  try:
    sea,rch = tuple(search.split(':'))
    if 'train' in sea:
      return st.dataframe(train.query(rch), use_container_width=True)
    elif 'test' in sea:
      return st.dataframe(test.query(rch), use_container_width=True)
    elif 'subm' in sea:
      return st.dataframe(subm.query(rch), use_container_width=True)
    else:
      st.write("Either Dtaframe is not there or queried column is not there")
  except SyntaxError:
    st.write('Syntax error Please check the syntax again')
  except ValueError:
    st.write('Please Enter any input then press enter')
query(search)
st.markdown('#')

#______________________________________________________________preprocessor setup_____________________________________________________________
uneccessary = st.multiselect('Uneccesary',[x for x in cols])
binning = st.multiselect('Binning',[x for x in cols if x not in uneccessary])
target = st.multiselect('Target',[x for x in cols if x not in uneccessary])
test_size = st.number_input('Test size',min_value=0.3,max_value=1.0,step=0.1)
# if st.button(label='Make', use_container_width=False):
X=pd.DataFrame()
y=pd.DataFrame()
if train.empty!=True:
  X = train.drop(uneccessary+target,axis=1)
  y = train[target]
  if X.empty!=True and y.empty!=True:
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
    st.write(f'X_train:{X_train.shape},\t\ty_train:{X_val.shape}')
    st.write(f'X_test:{y_train.shape},\t\ty_test:{y_val.shape}')
st.markdown('#')


def stnd(std_cols):
  #Standardization
  std_sc=StandardScaler()
  Tr_std=std_sc.fit_transform(X_train[std_cols])
  Va_std=std_sc.transform(X_val[std_cols])
  Tr_std=pd.DataFrame(Tr_std,columns=std_cols)
  Va_std=pd.DataFrame(Va_std,columns=std_cols)
  return [Tr_std,Va_std]

def mi_ma(mi_ma_cols):
  #Minmaxscaling
  mi_ma_sc = MinMaxScaler()
  Tr_mi_ma=mi_ma_sc.fit_transform(X_train[mi_ma_cols])
  Va_mi_ma=mi_ma_sc.transform(X_val[mi_ma_cols])
  Tr_mi_ma=pd.DataFrame(Tr_mi_ma,columns=mi_ma_cols)
  Va_mi_ma=pd.DataFrame(Va_mi_ma,columns=mi_ma_cols)
  return [Tr_mi_ma,Va_mi_ma]

def robu(robu_cols):
  #Robustscaling
  robu_sc = RobustScaler()
  Tr_robu=robu_sc.fit_transform(X_train[robu_cols])
  Va_robu=robu_sc.transform(X_val[robu_cols])
  Tr_robu=pd.DataFrame(Tr_robu,columns=robu_cols)
  Va_robu=pd.DataFrame(Va_robu,columns=robu_cols)
  return [Tr_robu,Va_robu]

def dmfc(dumm):
  #Dummification
  Tr_dmfc = pd.get_dummies(X_train[dumm])
  Va_dmfc = pd.get_dummies(X_val[dumm])
  return [Tr_dmfc,Va_dmfc]

def o_hot_enc(OHE_cols):
  #One Hot Encoding
  OHE = OneHotEncoder(handle_unknown='ignore')
  Tr_OHE=OHE.fit_transform(X_train[OHE_cols])
  Va_OHE=OHE.transform(X_val[OHE_cols])
  Tr_OHE=pd.DataFrame(Tr_OHE,columns=OHE_cols)
  Va_OHE=pd.DataFrame(Va_OHE,columns=OHE_cols)
  return [Tr_OHE,Va_OHE]


with st.form(key='preprocessing selector'):
  d1,d2,d3 = st.columns(3)
  with d1:
    for_num = st.multiselect('Scaler',['Standard','Minmax','Robust'],max_selections=2)
  with d2:
    for_cat = st.multiselect('Encoding',['Dummification','OneHotEnc','TargetMeanEnc','FrequencyEnc','A_LabelEnc'],max_selections=2)
  with d3:
    for_tar = st.multiselect('Target Encoding',['T_LabelEnc','BinaryEnc','OrdinalEnc'],max_selections=1)
  if st.form_submit_button(label = 'Preprocess', use_container_width=False):
    st.success('DONE')
st.markdown('#')

with st.form(key='preprocessing cols selector'):
  t1,t2,t3 = st.columns(3)
  with t1:
    if 'Standard' in for_num:
      std = st.multiselect('Stand-- cols',X.columns)
    if 'Minmax' in for_num:
      minmax = st.multiselect('MinMax-- cols',X.columns)
    if 'Robust' in for_num:
      Robust = st.multiselect('Robust-- cols',X.columns)
      
  with t2:
    if 'Dummification' in for_cat:
      dumm = st.multiselect('Dumm-- cols',X.columns)
    if 'OneHotEnc' in for_cat:
      OHE_cols = st.multiselect('OneHot-- cols',X.columns)
    if 'TargetMeanEnc' in for_cat:
      TarMeaEnc = st.multiselect('TargetMean-- cols',X.columns)
    if 'FrequencyEnc' in for_cat:
      freqEnc = st.multiselect('Frequency-- cols',X.columns)
    if 'A_LabelEnc' in for_cat:
      TarMeaEnc = st.multiselect('A_Label-- cols',X.columns)

  with t3:
    if 'T_LabelEnc' in for_tar:
      enc = st.multiselect('T_Label- cols',y.columns)
    if 'BinaryEnc' in for_tar:
      BinEnc = st.multiselect('Binary- cols',y.columns)
    if 'OrdinalEnc' in for_tar:
      OrdEnc = st.multiselect('Ordinal- cols',y.columns)
  if st.form_submit_button(label = 'DO', use_container_width=False):
    tr_std,va_std = stnd(std)
    tr_mi_ma,va_mi_ma = mi_ma(X_train[minmax])
    tr_robu,va_robu = robu(X_train[Robust])
    tr_dmfc,va_dmfc = dmfc(dumm)
    tr_OHE,va_OHE =o_hot_enc(OHE_cols)


# def prepro(val1=None,val2=None,val3=None):
#   lst2 = []
#   if 'Standard' in val1:
#     lst2.append('Standard()')
#   if 'Minmax' in val1:
#     lst2.append('MinmaxScalar()')
#   if 'Robust' in val1:
#     lst2.append('RobustScalar()')

#   if 'Dummification' in val2:
#     lst2.append('Dummification()')
#   if 'OneHotEnc' in val2:
#     lst2.append('OneHotEnc()')
#   if 'TargetMeanEnc' in val2:
#     lst2.append('TargetMeanEnc()')
#   if 'FrequencyEnc' in val2:
#     lst2.append('FrequencyEnc()')
#   if 'A_LabelEnc' in val2:
#     lst2.append('A_LabelEnc()')

#   if 'T_LabelEnc' in val3:
#     lst2.append('T_LabelEnc()')
#   elif 'BinaryEnc' in val3:
#     lst2.append('BinaryEnc()')     
#   elif 'OrdinalEnc' in val3:
#     lst2.append('OrdinalEnc()')   
#   return lst2
# with st.form(key='preprocessing cols selector'):
#   t1,t2,t3 = st.columns(3) 
#   colz=prepro(for_num,for_cat,for_tar)
#   if 'Standard()' in colz:
#     with t1:
#       std = st.multiselect('Stand-- cols',X.columns)
#       stnd(X_train[std])
#   if 'MinmaxScalar()' in colz:
#     with t1:
#       minmax = st.multiselect('MinMax-- cols',X.columns)
#       mi_ma(X_train[minmax])
#   if 'Robust()' in colz:
#     with t1:
#       Robust = st.multiselect('Robust-- cols',X.columns)
#       robu(X_train[Robust])

#   if 'Dummification()' in colz:
#     with t2:
#       dumm = st.multiselect('Dumm-- cols',X.columns)
#       pd.get_dummies(data=train)
#   if 'OneHotEnc()' in colz:
#     with t2:
#       OneHotEnc = st.multiselect('OneHot-- cols',X.columns)
#   if 'TargetMeanEnc()' in colz:
#     with t2:
#       TarMeaEnc = st.multiselect('TargetMean-- cols',X.columns)
#   if 'FrequencyEnc()' in colz:
#     with t2:
#       freqEnc = st.multiselect('Frequency-- cols',X.columns)
#   if 'A_LabelEnc()' in colz:
#     with t2:
#       TarMeaEnc = st.multiselect('A_Label-- cols',X.columns)

#   if 'T_LabelEnc()' in colz:
#     with t3:
#       enc = st.multiselect('T_Label- cols',y.columns)
#   if 'BinaryEnc()' in colz:
#     with t3:
#       BinEnc = st.multiselect('Binary- cols',y.columns)
#   if 'OrdinalEnc()' in colz:
#     with t3:
#       OrdEnc = st.multiselect('Ordinal- cols',y.columns)
#   if st.form_submit_button(label = 'Make', use_container_width=False):
#     st.success('DONE')



# d1,d2,d3 = st.columns(3)
# with d1:
#   for_num = st.multiselect('Scaler',['Standard','Minmax','Robust'],max_selections=2)
# with d2:
#   for_cat = st.multiselect('Encoding',['Dummification','OneHotEnc','TargetMeanEnc','FrequencyEnc','LabelEnc'],max_selections=2)
# with d3:
#   for_tar = st.multiselect('Target Encoding',['LabelEnc','BinaryEnc','OrdinalEnc'],max_selections=1)
# def prepro(val1,val2,val3):
#   lst2 = []
#   if val1==['Standard']:
#     lst2.append('Standard()')
#   if val1==['Minmax']:
#     lst2.append('MinmaxScalar()')
#   if val1==['Robust']:
#     lst2.append('RobustScalar()')

#   if val2 == ['Dummification']:
#     lst2.append('Dummification()')
#   if val2 == ['OneHotEnc']:
#     lst2.append('OneHotEnc()')
#   if val2 == ['TargetMeanEnc']:
#     lst2.append('TargetMeanEnc()')
#   if val2 == ['FrequencyEnc']:
#     lst2.append('FrequencyEnc()')
#   if val2 == ['LabelEnc']:
#     lst2.append('LabelEnc()')

#   if val3 == ['LabelEnc']:
#     lst2.append('Label_Enc()')
#   if val3 == ['BinaryEnc']:
#     lst2.append('BinaryEnc()')     
#   if val3 == ['OrdinalEnc']:
#     lst2.append('OrdinalEnc()')   
#   return lst2
# st.write(prepro(for_num,for_cat,for_tar))
# d4,d5,d6,d7,d8,d9 = st.columns(6) 
# colz=prepro(for_num,for_cat,for_tar)
# if 'Standard' in colz:
#   with d4:
#     std = st.multiselect('Std cols',X.columns)
# if 'MinmaxScalar' in colz:
#   with d5:
#     minmax = st.multiselect('MinMax cols',X.columns)
# if 'Robust' in colz:
#   with d6:
#     Robust = st.multiselect('Robust cols',X.columns)
# if 'Dummification' in colz:
#   with d7:
#     minmax = st.multiselect('Dumm cols',X.columns)
# if 'OneHotEnc' in colz:
#   with d8:
#     OneHotEnc = st.multiselect('OneHotEnc cols',X.columns)
# with d9:
#     enc = st.multiselect('Enc cols',X.columns)
# if st.button(label = 'Make1', use_container_width=True):
#   st.write('hello')





# with st.container():
#   d1,d2,d3 = st.columns(3)
#   with d1:
#     for_num = st.multiselect('Scaler',['Standard','Minmax','Robust'],max_selections=2)
#   with d2:
#     for_cat = st.multiselect('Encoding',['Dummification','OneHotEnc','TargetMeanEnc','FrequencyEnc','LabelEnc'],max_selections=2)
#   with d3:
#     for_tar = st.multiselect('Target Encoding',['LabelEnc','BinaryEnc','OrdinalEnc'],max_selections=1)
#   if st.button(label = 'Preprocess', use_container_width=True):
#     def prepro(val1,val2,val3):
#       lst2 = []
#       if val1==['Standard']:
#         lst2.append('Standard()')
#       if val1==['Minmax']:
#         lst2.append('MinmaxScalar()')
#       if val1==['Robust']:
#         lst2.append('RobustScalar()')

#       if val2 == ['Dummification']:
#         lst2.append('Dummification()')
#       if val2 == ['OneHotEnc']:
#         lst2.append('OneHotEnc()')
#       if val2 == ['TargetMeanEnc']:
#         lst2.append('TargetMeanEnc()')
#       if val2 == ['FrequencyEnc']:
#         lst2.append('FrequencyEnc()')
#       if val2 == ['LabelEnc']:
#         lst2.append('LabelEnc()')

#       if val3 == ['LabelEnc']:
#         lst2.append('Label_Enc()')
#       if val3 == ['BinaryEnc']:
#         lst2.append('BinaryEnc()')     
#       if val3 == ['OrdinalEnc']:
#         lst2.append('OrdinalEnc()')   
#       return lst2
#     st.write(prepro(for_num,for_cat,for_tar))
# with st.container():
#   d4,d5,d6,d7,d8,d9 = st.columns(6) 
#   # colz=prepro(val1,val2,val3)
#   colz=['Standard','MinmaxScalar','Robust','Dummification']
#   if 'Standard' in colz:
#     with d4:
#       std = st.multiselect('Std cols',X.columns)
#   if 'MinmaxScalar' in colz:
#     with d5:
#       minmax = st.multiselect('MinMax cols',X.columns)
#   if 'Robust' in colz:
#     with d6:
#       Robust = st.multiselect('Robust cols',X.columns)
#   if 'Dummification' in colz:
#     with d7:
#       minmax = st.multiselect('Dumm cols',X.columns)
#   if 'OneHotEnc' in colz:
#     with d8:
#       OneHotEnc = st.multiselect('OneHotEnc cols',X.columns)
#   with d9:
#       enc = st.multiselect('Enc cols',X.columns)
#   if st.button(label = 'Make1', use_container_width=True):
#     st.write('hello')




# col1,col2=st.columns(2)
# with col1:
#   s1 = st.multiselect('S1',[1,2,3])

# with col2:
#   s2 = st.multiselect('S2',[1,2,3])

# with col1:
#   ss1 = st.multiselect('SS1',[1,2,3])

# with col2:
#    ss2 = st.multiselect('SS2',[1,2,3])



# with st.form(key='columns spliter'):
#     c1,c2=st.columns(2)
#     with c1:
#         uneccessary = st.multiselect('Select uneccessary columns',[x for x in cols])
#     with c2:
#         target = st.multiselect('Target',[x for x in cols if x not in uneccessary])
#     submit_button = st.form_submit_button(label='Take')







# with st.form(key='columns_in_form1'):
#     d1, d2, d3, d4 = st.columns(4)
#     with d1:
#         CustomerID = st.text_input("CustomerID",value='Cust10000'	)
#     with d2:
#         DateOfIncident = st.text_input("DateOfIncident",value='2015-02-03')
#     with d3:
#         TypeOfIncident = st.text_input("TypeOfIncident",value='Multi-vehicle Collision')
#     with d4:
#         TypeOfCollission = st.text_input("TypeOfCollission",value='Side Collision')
#     submitButton = st.form_submit_button(label = 'Submit1')
# with st.form(key='columns selector'):
#     d1,d2,d3 = st.columns(3)
#     cols = train.columns
#     with d1:
#         ls1 = cols
#         num_opt = st.multiselect('What are your numerical columns',ls1)
#     with d2:
#         ls2 = [x for x in ls1 if x not in num_opt]
#         cat_opt = st.multiselect('What are your categorical columns',ls2)
#     with d3:
#         ls3 = [x for x in ls2 if x not in num_opt+cat_opt]
#         date_opt = st.multiselect('What are your datetime columns',ls3)

# import streamlit as st
# import pandas as pd
# # Function 
# def color_df(val):
# if val > 21:
#     color = 'green'
# else :
#    color = 'red'
# return f'background-color: {color}'
# # Our example Dataset
# data = [['Tom', 23], ['Nick', 18], ['Bob', 20], ['Martin', 25]]
# # Create Pandas DataFrame
# df = pd.DataFrame(data, columns = ['Name', 'Age'])
 
# # Using Style for the Dataframe
# st.dataframe(df.style.applymap(color_df, subset=['Age']))





# import streamlit as st

# col1, col2 = st.columns(2)

# Keep the list in the session state
# if 'myList' not in st.session_state: 
#     st.session_state['myList'] = []

# symbols_list = st.session_state['myList']

# def add_symbols(tckr):
#     symbols_list.append(tckr)
#     col2.write(symbols_list)

# # Reset the session state 
# def empty_list(): del st.session_state['myList']

# with col1:
#     st.button(label="MYTIL", on_click=add_symbols, args=["MYTIL.AT"])
#     st.button(label="OPAP", on_click=add_symbols, args=["OPAP.AT"])
#     st.button(label="ADMIE", on_click=add_symbols, args=["ADMIE.AT"])
#     st.button(label="EYDAP", on_click=add_symbols, args=["EYDAP.AT"])
#     st.button(label="ELPE", on_click=add_symbols, args=["ELPE.AT"])
#     st.button(label="HTO", on_click=add_symbols, args=["HTO.AT"])

#     st.button(label="❌", on_click=empty_list)



# is st.button('Load Data') or st.session_state.load_state:
# st.session_state.load_state == True
# _df = pd.DataFrame(_dict)


# --- Initialising SessionState ---
# if "load_state" not in st.session_state:
#      st.session_state.load_state = True

    # if 'sheet' in uploaded_file.type:
    #   if 'train' in uploaded_file.name:
    #     st.write('Train Data')
    #     train = pd.read_excel(uploaded_file)
    #     st.dataframe(train, use_container_width=True)
    #   if 'test' in uploaded_file.name:
    #     st.write('Test Data')
    #     test = pd.read_excel(uploaded_file)
    #     st.dataframe(test, use_container_width=True)
    #   if 'subm' in uploaded_file.name:
    #     st.write('submission Data')
    #     subm = pd.read_excel(uploaded_file)
    #     st.dataframe(subm, use_container_width=True)
    #   else:
    #     st.write('Train Data')
    #     st.write(uploaded_file.name[:-5])
    #     train = pd.read_excel(uploaded_file.name[:-5])
    #     st.dataframe(train, use_container_width=True)
