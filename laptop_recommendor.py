import streamlit as st
import pickle
import numpy as np
# from sklearn import * 

#import the model 
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

#brand 
company = st.selectbox('Brand', df['Company'].unique())

#Type of laptop
Type = st.selectbox('Type', df['TypeName'].unique())

#Ram
Ram = st.selectbox('Ram(in GB)', [2,4,6,8,12,16,24,32,64])

#weight
weight = st.number_input('Weight of the laptop')

#TouchScreen
Touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

#IPS
IPS = st.selectbox('IPS', ['No', 'Yes'])

#screensize
ScreenSize = st.number_input('Screen Size')

#resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

hdd = st.selectbox('HDD', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD', [0, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu_brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    #query
    ppi = None
    if Touchscreen == 'Yes':
        Touchscreen = 1 
    else:
        Touchscreen = 0

    if IPS == 'Yes':
        IPS = 1
    else:
        IPS = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/ScreenSize 

    query = np.array([company, Type, Ram, weight, Touchscreen, IPS, 
                      ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1,12)
    # st.title(str(int(np.exp(pipe.predict(query)[0]))))
    st.title("The predicted price of this configuration is ₹" + str(int(np.exp(pipe.predict(query)[0]))))
