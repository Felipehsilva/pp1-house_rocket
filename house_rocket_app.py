# ------------------------------
# how to run streamlit
# ------------------------------
# streamlit hello
# streamlit run house_rocket_app.py

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px

st.title( 'House Rocket Company' )
st.markdown( 'Welcome to House Rocket Data Analysis' )

st.header( 'Load data' )

# read data
@st.cache_data(  ) #st.cache_data is the recommended way to cache computations that return data
def get_data( path ):
    data = pd.read_csv( path )
    data['date'] = pd.to_datetime( data['date'] )
    return data

# load data
data = get_data( 'datasets/kc_house_data.csv' )

# filter bedrooms
bedrooms = st.sidebar.multiselect(
    'Number of Bedrooms', 
    data['bedrooms'].unique() )

if bedrooms:
    selected_bedrooms = ', '.join(map(str, sorted(bedrooms)))
    st.write('You choose:', selected_bedrooms)
    df = data[data['bedrooms'].isin(bedrooms)]
else:
    st.write('None filter selected.')
    df = data.copy()

df = data[data['bedrooms'].isin(bedrooms)]
st.dataframe( df.head() )

# data dimension
st.write( 'Number of Rows:', data.shape[0] )
st.write( 'Number of Cols:', data.shape[1] )

# data types
st.header( 'Data Types' )
st.write( data.dtypes )

# data descriptive
num_attributes = data.select_dtypes(include=['int64','float64'])

# central tendency
media = pd.DataFrame( num_attributes.apply( np.mean ) )
mediana = pd.DataFrame( num_attributes.apply( np.median ) )
std = pd.DataFrame( num_attributes.apply( np.std ) )

# dispersion
std = pd.DataFrame( num_attributes.apply( np.std ) )
max_ = pd.DataFrame( num_attributes.apply( np.max ) )
min_ = pd.DataFrame( num_attributes.apply( np.min ) )

df1 = pd.concat([max_, min_, media, mediana, std],axis=1).reset_index()
df1.columns = ['attributes', 'mean', 'median', 'std', 'max', 'min']
st.header( 'Data Descriptive' )
st.dataframe( df1 )

#data = data.rename( columns={'long':'lon'} )
#houses = data[['id', 'lat', 'lon', 'price']]
#
#st.map( houses )

# ---------------------------------------------------------------------------------------
#                                           MAP
# -------------------------------------------------------------------------------------
# define level of prices
for i in range( len( data ) ):
    if data.loc[i, 'price'] <= 321950:
        data.loc[i, 'level'] = 0

    elif ( data.loc[i,'price'] > 321950 ) & ( data.loc[i,'price'] <= 450000 ):
        data.loc[i, 'level'] = 1

    elif ( data.loc[i,'price'] > 450000 ) & ( data.loc[i,'price'] <= 645000 ):
        data.loc[i, 'level'] = 2

    else:
        data.loc[i, 'level'] = 3



# plot map
st.title( 'House Rocket Map' )
is_check = st.checkbox( 'Display Map')

# filters
price_min = int( data['price'].min() )
price_max = int( data['price'].max() )
price_avg = int( data['price'].median() )
price_slider = st.slider('Price Range', price_min, price_max, price_avg)

if is_check:
    # select rows
    houses = data[data['price'] < price_slider][['id','lat','long',
                                                 'price', 'level']]

    # draw map
    fig = px.scatter_map( 
        houses, 
        lat="lat", 
        lon="long", 
        color="level", 
        size="price",
        color_continuous_scale=px.colors.cyclical.IceFire, 
        size_max=15, 
        zoom=10 )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart( fig )
