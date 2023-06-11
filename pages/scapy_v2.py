import streamlit as st
# import pandas as pd
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report
# import seaborn as sns
# import matplotlib.pyplot as plt

import pymongo
import pandas as pd
import numpy as np

# from pandas.api.types import (
#     is_categorical_dtype,
#     is_datetime64_any_dtype,
#     is_numeric_dtype,
#     is_object_dtype,
# )

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import urllib.parse

import seaborn as sns
# cm = sns.diverging_palette(5, 133, sep=80, n=7, as_cmap=True)
# cm = sns.color_palette("blend:White,green", as_cmap=True)



usr = st.secrets["usr"]
pwd = st.secrets["pwd"]
# username = urllib.parse.quote_plus(usr)
# password = urllib.parse.quote_plus(pwd)

# Define a function for 
# colouring negative values 
# red and positive values black
st.title('Scalping  :blue[Analysis] _Dashboard_ :sunglasses:')

def highlight_max(cell):
    if type(cell) != str and cell < 0 :
        return 'color: red'
    else:
        return 'color: green'

# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection(usr,pwd):
    username = urllib.parse.quote_plus(usr)
    password = urllib.parse.quote_plus(pwd)
    uri = "mongodb+srv://%s:%s@cluster0.tajveiy.mongodb.net/?retryWrites=true&w=majority" % (username, password)
    client = MongoClient(uri, server_api=ServerApi('1'))
    return client
    # return pymongo.MongoClient(**st.secrets["mongo"])

client = init_connection(usr,pwd)

@st.cache_data(ttl=600)
def get_data():
    db = client.Scalping_data
    items = db.d2.find()
    if len(list(items))==0:
        items = db.d1.find()
    items = list(items)  # make hashable for st.cache_data
    return pd.DataFrame(items)


df = get_data()

df["Datetime"] = pd.to_datetime(df["Date"])
df['Hour'] = df['Datetime'].apply(lambda x: x.hour)
df['Day'] = df['Datetime'].apply(lambda x: x.day_name())
df['Date'] = df['Datetime'].apply(lambda x: x.date())
df["pnl"] = ((df["sell_at"]-df["buy_at"])/df["buy_at"])*100


df1 = df[df["note"] == "stoploss_hit"]
df1 = df1.sort_values("Datetime")
df1["running_pnl"] = df1.groupby('Date').pnl.cumsum()

pivot = np.round(pd.pivot_table(df1, values='pnl', 
                                index=['Date'], 
                                columns=['Day'], 
                                aggfunc=np.sum,
                                fill_value=0),2)

pivot1 = np.round(pd.pivot_table(df1, values='pnl', 
                                index=['Date'], 
                                columns=['Hour'], 
                                aggfunc=np.sum,
                                fill_value=0),2)

pivot2 = np.round(pd.pivot_table(df1, values='pnl', 
                                index=['Day'], 
                                columns=['Hour'], 
                                aggfunc=np.sum,
                                fill_value=0),2)

pivot3 = np.round(pd.pivot_table(df1, values='pnl', 
                                index=['Date'], 
                                columns=['Hour'], 
                                aggfunc='count',
                                fill_value=0),2)
# Print results.

@st.cache_data
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv(index=False).encode('utf-8')


st.header('Sample Data')
st.write(df1)
st.download_button(
  label="Download data as CSV",
  data=convert_df_to_csv(df1),
  file_name='df.csv',
  mime='text/csv',
)

st.subheader('Date vs Day PNL')
st.write(pivot.style.applymap(highlight_max))
st.subheader('Date vs Hour PNL')
st.write(pivot1.style.applymap(highlight_max))
st.subheader('Day vs Hour PNL')
st.write(pivot2.style.applymap(highlight_max))
st.subheader('Date vs Hour Trades Count')
st.write(pivot3)





# with st.sidebar:
#     with st.form(key='my_form', clear_on_submit=True):
#         uploaded_f = st.file_uploader("FILE UPLOADER")
#         submitted = st.form_submit_button("UPLOAD!")
        
#         if submitted and uploaded_f is not None:
#             st.write("UPLOADED!")
#             # do stuff with your uploaded file



# @st.cache
# def load_data1(uploaded_f):
#     data1 = pd.read_csv(uploaded_f)
#     return data1


# @st.experimental_memo(ttl=24 * 3600, persist="disk", show_spinner=False)
# def profiling(df):
#     pr = df.profile_report()
#      # This makes the function take 2s to run
#     return pr


# if uploaded_f is not None:
#     df = load_data1(uploaded_f)
#     st.write(df.head())
#     pr = profiling(df)
#     st_profile_report(pr)

