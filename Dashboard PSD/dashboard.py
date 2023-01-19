# how to streamlit run /home/noname00/Documents/Dashboard_PSD/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import lazypredict
import numpy as np
import urllib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn import metrics


dir="".join(["/home/noname00/Documents/Dashboard_PSD/dataset/SaYoPillow.csv"])
df=pd.read_csv(dir)
st.markdown("<h1 style='text-align: center; color: Black;'> Deteksi Stres Manusia Pada Saat Tidur</h1>", unsafe_allow_html=True)

st.dataframe(df,900,500)
# st.markdown(""" 
# *   sr = Snoring Rate (Tingkat dengkuran pengguna)
# *   rr = Respiration Rate (Tingkat pernapasan)
# *   t = Body Temperature (Suhu tubuh)
# *   lm = Limb Movement Rate (Laju pergerakan tungkai)
# *   bo = Blood Oxygen Level (Kadar Oksigen di Darah)
# *   rem  = Eye Movement (Pergerakan Mata)
# *   sr.1 = Number of Hours of Sleep (Jumlah jam tidur)
# *   hr = Heart Rate (Detak Jantung)
# *   sl - Stress Levels/Tingkat Stress (0-low/normal, 1-medium low, 2-medium, 3-medium high, 4-high)
# """)

colA, colB= st.columns(2)
colA.metric("Jumlah Data", df.shape[0])
colB.metric("Jumlah Fitur", df.shape[1])

col1,col2=st.columns(2)

with col1:
  st.header("Distribusi data")
  color = '#eab889'
  data = df.copy()
  data.drop('sl', axis = 1, inplace = True)
  data.hist(bins=15,figsize=(25,15),color=color)
  plt.rcParams['font.size'] = 18
  st.pyplot(plt)

  

with col2:
  st.header("Hubungan Stress level dengan fitur")
  fig = plt.figure(figsize=(40, 15))
  rows = 2
  columns = 4
  tmp=["stress level"]
  for i in range(len(df.columns[:-1])):
    fig.add_subplot(rows, columns, (i+1))
    img = sns.pointplot(x=df.columns[-1],y=df.columns[i],data=df,color='lime')
  st.pyplot(fig)
st.markdown("<h1 style='text-align: center; color: Black;'> Menampilkan data setiap kelas </h1>", unsafe_allow_html=True)
# st.header("Menampilkan data setiap kelas")
fig = px.histogram(df, x="sl",color="sl")
fig.update_layout(bargap=0.1)
st.plotly_chart(fig)

# st.markdown("<h1 style='text-align: center; color: Black;'> Data Preparation</h1>", unsafe_allow_html=True)
# st.header("1) Mengubah nama kolom agar mudah dibaca")
df.columns = ['snoring_rate', 'respiration_rate', 'temperature', 'limb_move', 'blood_oxygen', 'eye_move', 'sleep_hour', 'heart_rate', 'stress_level']
# st.dataframe(df)

# st.header("2) Memisahkan dataframe fitur dan dataframe kelas")
x = df.copy()
x.drop('stress_level', axis = 1, inplace = True)
y = df['stress_level']
# st.dataframe(x)
# st.caption("dataframe fitur")
# st.dataframe(y)
# st.caption("dataframe kelas")

# st.header("3) Melakukan normalisasi Min-Max pada dataframe fitur")
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
# tmp_df=pd.DataFrame()
# tmp_df[['snoring_rate', 'respiration_rate', 'temperature', 'limb_move', 'blood_oxygen', 'eye_move', 'sleep_hour', 'heart_rate']]=x
# st.dataframe(tmp_df)

# st.header("4) Memisahkan data training dan testing dengan perbandingan 80%:20%")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2,random_state =123)

# st.subheader("- Data training terdapat {} data".format(len(X_train)))
# st.subheader("- Data testing terdapat {} data".format(len(X_test)))
# tmp_df=pd.DataFrame()
# tmp_df[['snoring_rate', 'respiration_rate', 'temperature', 'limb_move', 'blood_oxygen', 'eye_move', 'sleep_hour', 'heart_rate']]=X_train
# st.dataframe(tmp_df)


st.markdown("<h1 style='text-align: center; color: Black;'> Data Modelling</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
   st.header("KNN")
   model_KNN = KNeighborsClassifier(n_neighbors=5)
   model_KNN.fit(X_train, y_train)
   predict_KNN = model_KNN.predict(X_test)
   confusion_matrix = metrics.confusion_matrix(y_test, predict_KNN)
   cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4])
   cm_display.plot()
   st.pyplot(plt)
   st.caption("Akurasi {} %".format(accuracy_score(y_test,predict_KNN)*100))


with col2:
   st.header("Naive Bayes")
   model_NB = GaussianNB()
   model_NB.fit(X_train, y_train)
   predict_NB = model_NB.predict(X_test)
   confusion_matrix = metrics.confusion_matrix(y_test, predict_NB)
   cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4])
   cm_display.plot()
   st.pyplot(plt)
   st.caption("Akurasi {} %".format(accuracy_score(y_test,predict_NB)*100))


with col3:
   st.header("Decision Tree")
   model_DT = DecisionTreeClassifier(max_depth=40,random_state=101)
   model_DT.fit(X_train, y_train)
   predict_DT = model_DT.predict(X_test)
   confusion_matrix = metrics.confusion_matrix(y_test, predict_DT)
   cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4])
   cm_display.plot()
   st.pyplot(plt)
   st.caption("Akurasi {:02f} %".format(accuracy_score(y_test,predict_DT)*100))
   
st.markdown("<h1 style='text-align: center; color: Black;'> DEMO</h1>", unsafe_allow_html=True)
sr = st.text_input('Snoring rate')
rr = st.text_input('Respiration rate')
t = st.text_input('Body Temperature')
lm = st.text_input('Limb Movement Rate')
bo = st.text_input('Blood Oxygen Level')
rem = st.text_input('Eye Movement')
sr1 = st.text_input('Number Hours of Sleep')
hr = st.text_input('Heart Rate')
algoritm = st.selectbox(
        "model",
        ("KNN", "Naive Bayes", "Decision Tree"))
if st.button("Predict"):
    data_input=np.array([float(sr),float(rr),float(t),float(lm),float(bo),float(rem),float(sr1),float(hr)]).reshape(1, -1)
    data_input=scaler.transform(data_input)
    if algoritm=="KNN":
      result=model_KNN.predict(data_input)
      st.subheader("Stress Level : {}".format(result[0]))
    elif algoritm=="Naive Bayes":
      result=model_NB.predict(data_input)
      st.subheader("Stress Level : {}".format(result[0]))
    else:
      result=model_DT.predict(data_input)
      st.subheader("Stress Level : {}".format(result[0]))
    