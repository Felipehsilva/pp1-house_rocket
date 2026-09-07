#to run in terminal execute -> streamlit run streamlit_final.py

import geopandas
import streamlit as st
import pandas    as pd
import numpy     as np
import folium
import requests
from datetime import datetime, time, date
from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster
import plotly.express as px 
from folium.plugins import HeatMap
import joblib
import os

MODEL_PATH = 'models/house_price_random_forest_compact_pipeline.joblib'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'models/house_price_random_forest_pipeline.joblib'

@st.cache_resource
def load_house_price_model():
    return joblib.load(MODEL_PATH)

# ------------------------------------------
# settings
# ------------------------------------------
#use the full space for the page
st.set_page_config( layout='wide' )


# ------------------------------------------
# Helper Functions
# ------------------------------------------
@st.cache_data(  )
def get_data( path ):
    data = pd.read_csv( path )

    return data

def plotly_chart(fig):
    st.plotly_chart(fig, config={"responsive": True})


@st.cache_data(  )
def get_geofile(path_json):
    geofile = geopandas.read_file(path_json)
    #Changing the name from JSON to ZIP
    geofile = geofile.rename(columns={'ZCTA5CE10': 'ZIP'})
    geofile['ZIP'] = geofile['ZIP'].astype(str)
    return geofile    



def set_attributes( data ):
    data['price_m2'] = data['price'] / data['sqft_lot'] 

    return data


def data_overview( data ):
    f_attributes = st.sidebar.multiselect( 'Digite as colunas', data.columns ) 
    f_zipcode = st.sidebar.multiselect( 'Digite o código postal', data['zipcode'].unique() )

    st.title( 'Visão geral dos dados')
    #Select columns and zip
    if ( f_zipcode != [] ) & ( f_attributes != [] ):
        data = data.loc[data['zipcode'].isin( f_zipcode ), f_attributes]
    #Select only ZIP
    elif ( f_zipcode != [] ) & ( f_attributes == [] ):
        data = data.loc[data['zipcode'].isin( f_zipcode ), :]
    #Select only columns(DF attributes)
    elif ( f_zipcode == [] ) & ( f_attributes != [] ):
        data = data.loc[:, f_attributes]

    else:
        data = data.copy()

    st.write( data.head() )

    c1, c2 = st.columns((1, 1) )  # esses numeros  da tupla (2,1) indicam o quanto de espaco eu quero em cada coluna, no caso a coluna 1 seria mais larga que a segunda (1,1) indica larguras iguais 

    # Average metrics
    df1 = data[['id', 'zipcode']].groupby( 'zipcode' ).count().reset_index()
    df2 = data[['price', 'zipcode']].groupby( 'zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby( 'zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby( 'zipcode').mean().reset_index()


    # merge
    m1 = pd.merge( df1, df2, on='zipcode', how='inner' )
    m2 = pd.merge( m1, df3, on='zipcode', how='inner' )
    df = pd.merge( m2, df4, on='zipcode', how='inner' )

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

    c1.header( 'Valores Médios' )
    c1.dataframe( df, height=600 )

    # Statistic Descriptive
    num_attributes = data.select_dtypes( include=['int64', 'float64'] )
    media = pd.DataFrame( num_attributes.apply( np.mean ) )
    mediana = pd.DataFrame( num_attributes.apply( np.median ) )
    std = pd.DataFrame( num_attributes.apply( np.std ) )

    max_ = pd.DataFrame( num_attributes.apply( np.max ) ) 
    min_ = pd.DataFrame( num_attributes.apply( np.min ) ) 

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1 ).reset_index()
    df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std'] 

    c2.header( 'Análise Descritiva' )
    c2.dataframe( df1, height=800 )

    return None


def region_overview( data, geofile ):
    st.title( 'Region Overview' )

    c1, c2 = st.columns( ( 1, 1 ) )
    c1.header( 'Densidade do Portfólio' )
    #Getting only the sample below in order to not break Streamlit load and render
    df = data.sample( 100 )
    
   

    # Base Map - Folium 
    density_map = folium.Map( location=[data['lat'].mean(), data['long'].mean() ],
                              default_zoom_start=10 )#initial zoom in themap 

    marker_cluster = MarkerCluster().add_to( density_map )
    for name, row in df.iterrows():
        folium.Marker( [row['lat'], row['long'] ], 
            popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( row['price'], 
                           row['date'], 
                           row['sqft_living'],
                           row['bedrooms'],
                           row['bathrooms'],
                           row['yr_built'] ) ).add_to( marker_cluster )


    with c1: # tudo que está dentro do with será renderizado dentro da coluna c1, uso with c1 por que o folium_static nao eh nativo do streamlit
        folium_static( density_map ) # folium_static lib que permite renderizar mapas do folium(HTML/JS) dentro do streamlit


    
    # # we don't have the same ZIP in JSON as we have in CSV, that is why this function does not work 
    #     # Region Price Map
    # c2.header( 'Price Density' )

    # df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    # df.columns = ['ZIP', 'PRICE']

    # geofile = geofile[geofile['ZIP'].isin( df['ZIP'].tolist() )]

    # region_price_map = folium.Map( location=[data['lat'].mean(), 
    #                                data['long'].mean() ],
    #                                default_zoom_start=15 ) 

    # common = set(df['ZIP']).intersection(set(geofile['ZIP']))
    # st.write(f'ZIPs em comum: {len(common)}')

    # df = df[df['ZIP'].isin(common)]
    # geofile = geofile[geofile['ZIP'].isin(common)]  
    # folium.Choropleth(
    #                     geo_data=geofile,
    #                     data=df,
    #                     columns=['ZIP', 'PRICE'],
    #                     key_on='feature.properties.ZIP',
    #                     fill_color='YlOrRd',
    #                     fill_opacity=0.7,
    #                     line_opacity=0.2,
    #                     legend_name='AVG PRICE'
    #                  ).add_to(region_price_map)


    # #with c2:
    #  #   folium_static( region_price_map )
    

    c2.header('Densidade de Preço (Mapa de Calor)')
    region_price_map = folium.Map(
    location=[data['lat'].mean(), data['long'].mean()],
    zoom_start=11)
    heat_data = data[['lat', 'long', 'price']].dropna()
    heat_data['price_norm'] = heat_data['price'] / heat_data['price'].max()
    HeatMap(
    data=heat_data[['lat', 'long', 'price_norm']].values.tolist(),
    radius=12,
    blur=15,
    max_zoom=13).add_to(region_price_map)
    with c2:
        folium_static(region_price_map)

    
    return None


def set_commercial( data ):
    st.sidebar.title( 'Opções Comerciais' )
    st.title( 'Atributos Comerciais' )

    # ---------- Average Price per year built
    # setup filters
    min_year_built = int( data['yr_built'].min() )
    max_year_built = int( data['yr_built'].max() )

    st.sidebar.subheader( 'Selecionar Ano Máximo de Construção' )
    f_year_built = st.sidebar.slider( 'Ano de Construção', min_year_built, max_year_built, min_year_built )

    st.header( 'Preço médio por ano de construção' )

    # get data
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d' )

    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()

    fig = px.line( df, x='yr_built', y='price' )
    st.plotly_chart( fig )


    # ---------- Average Price per day ------------------
    st.header( 'Preço Médio por Dia' )
    st.sidebar.subheader( 'Selecionar Data Máxima' )

    # setup filters
    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' )
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )

    f_date = st.sidebar.slider( 'Date', min_date, max_date, min_date )

    # filter data
    data['date'] = pd.to_datetime( data['date'] )
    df = data[data['date'] < f_date]
    df = df[['date', 'price']].groupby( 'date' ).mean().reset_index()

    fig = px.line( df, x='date', y='price' )
    st.plotly_chart( fig)

    # ---------- Histogram -----------
    st.header( 'Distribuição de Preços' )
    st.sidebar.subheader( 'Selecionar Preço Máximo' )

    # filters
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int( data['price'].mean() )

    f_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )

    df = data[data['price'] < f_price]

    fig = px.histogram( df, x='price', nbins=50 )
    st.plotly_chart( fig)

    return None


def set_phisical( data ):
    st.sidebar.title( 'Opções de Atributos' )
    st.title( 'Atributos da Casa' )

    # filters
    f_bedrooms = st.sidebar.selectbox( 'Número máximo de quartos', 
                                        sorted( set( data['bedrooms'].unique() ) ) )
    f_bathrooms = st.sidebar.selectbox( 'Número máximo de banheiros', 
                                        sorted( set( data['bathrooms'].unique() ) ) )

    c1, c2 = st.columns( 2 )

    # Houses per bedrooms
    c1.header( 'Casas por quartos' )
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram( df, x='bedrooms', nbins=19 )
    c1.plotly_chart( fig )

    # Houses per bathrooms
    c2.header( 'Casas por banheiros' )
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram( df, x='bathrooms', nbins=10 )
    c2.plotly_chart( fig )

    # filters
    f_floors = st.sidebar.selectbox('Número máximo de andares', sorted( set( data['floors'].unique() ) ) )
    f_waterview = st.sidebar.checkbox('Apenas Casas com Vista para a Água' )

    c1, c2 = st.columns( 2 ) #  Cria Duas colunas 50%/50% eh parecido com st.columns((1, 1)), porem essa segunda me permite customizar o tamanho de cada coluna

    # Houses per floors
    c1.header( 'Casas por andares' )
    df = data[data['floors'] < f_floors]
    fig = px.histogram( df, x='floors', nbins=19 )
    c1.plotly_chart( fig )

    # Houses per water view
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram( df, x='waterfront', nbins=10 )
    c2.header( 'Casas por vista para a água' )
    c2.plotly_chart( fig )

    return None


if __name__ == "__main__":
    # ETL
    path = 'datasets/kc_house_data.csv'
    #url='https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson' #not working
    path_json = "datasets/wa_washington_zip_codes_geo.min.json"


    # load data
    data = get_data( path )
    geofile = get_geofile( path_json )

    # transform data
    data = set_attributes( data )

    data_overview( data )

    region_overview( data, geofile )

    set_commercial( data )
    
    set_phisical( data )

    # ------------------------------------------
    # Previsao de Preco de Venda
    # ------------------------------------------
    st.divider()
    st.header('Previsão de Preço de Venda')
    st.caption(
        'Informe as características do imóvel para obter uma estimativa de preço '
        'com o modelo Random Forest treinado na etapa de Machine Learning.'
    )

    try:
        house_price_model = load_house_price_model()
        model_loaded = True
    except Exception as e:
        st.error(f'Erro ao carregar o modelo de previsão: {e}')
        model_loaded = False

    if model_loaded:
        with st.form('house_price_prediction_form'):
            c1, c2, c3 = st.columns(3)

            with c1:
                sale_date = st.date_input(
                    'Data estimada da venda',
                    value=date.today()
                )

                bedrooms = st.number_input(
                    'Quartos',
                    min_value=1,
                    max_value=20,
                    value=3,
                    step=1
                )

                sqft_living = st.number_input(
                    'Área habitável (sqft)',
                    min_value=1,
                    value=1800,
                    step=50
                )

                floors = st.selectbox(
                    'Número de andares',
                    options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                    index=0
                )

                view = st.selectbox(
                    'Nível de vista',
                    options=[0, 1, 2, 3, 4],
                    index=0
                )

            with c2:
                yrbuilt = st.number_input(
                    'Ano de construção',
                    min_value=1900,
                    max_value=sale_date.year,
                    value=2000,
                    step=1
                )

                bathrooms = st.number_input(
                    'Banheiros',
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.25
                )

                sqft_lot = st.number_input(
                    'Área do terreno (sqft)',
                    min_value=1,
                    value=6000,
                    step=100
                )

                waterfront = st.selectbox(
                    'Possui vista para água?',
                    options=[0, 1],
                    format_func=lambda value: 'Sim' if value == 1 else 'Não'
                )

                condition = st.selectbox(
                    'Condição do imóvel',
                    options=[1, 2, 3, 4, 5],
                    index=2,
                    help='1 representa condição mais baixa e 5 representa condição mais alta.'
                )

            with c3:
                yrrenovated = st.number_input(
                    'Ano da última reforma (0 se nunca foi reformado)',
                    min_value=0,
                    max_value=sale_date.year,
                    value=0,
                    step=1
                )

                grade = st.selectbox(
                    'Qualidade / grade do imóvel',
                    options=list(range(1, 14)),
                    index=6,
                    help='1 representa qualidade mais baixa e 13 representa qualidade mais alta.'
                )

                sqft_above = st.number_input(
                    'Área acima do solo (sqft)',
                    min_value=0,
                    value=1800,
                    step=50
                )

                sqft_basement = st.number_input(
                    'Área do porão (sqft)',
                    min_value=0,
                    value=0,
                    step=50
                )

                zipcode = st.number_input(
                    'CEP / Zipcode',
                    min_value=10000,
                    max_value=99999,
                    value=98178,
                    step=1
                )

                latitude = st.number_input(
                    'Latitude',
                    min_value=47.0,
                    max_value=48.0,
                    value=47.51,
                    step=0.01,
                    format='%.5f'
                )

                longitude = st.number_input(
                    'Longitude',
                    min_value=-123.0,
                    max_value=-121.0,
                    value=-122.26,
                    step=0.01,
                    format='%.5f'
                )

            submitted = st.form_submit_button('Prever preço de venda')

        if submitted:
            # Validacoes dos dados de entrada
            is_valid = True
            if sqft_living <= 0:
                st.warning('A área habitável (sqft_living) deve ser maior que zero.')
                is_valid = False
            if sqft_lot <= 0:
                st.warning('A área do terreno (sqft_lot) deve ser maior que zero.')
                is_valid = False
            if bedrooms <= 0:
                st.warning('O número de quartos (bedrooms) deve ser maior que zero.')
                is_valid = False
            if sqft_above > sqft_living:
                st.warning('A área acima do solo (sqft_above) não pode ser maior que a área habitável (sqft_living).')
                is_valid = False
            if sqft_basement > sqft_living:
                st.warning('A área do porão (sqft_basement) não pode ser maior que a área habitável (sqft_living).')
                is_valid = False
            if (sqft_above + sqft_basement) > sqft_living:
                st.warning('A soma da área acima do solo e do porão não pode ser maior que a área habitável (sqft_living).')
                is_valid = False
            if len(str(int(zipcode))) != 5:
                st.warning('O CEP (zipcode) deve possuir cinco dígitos.')
                is_valid = False
            if yrbuilt > sale_date.year:
                st.warning('O ano de construção (yrbuilt) não pode ser maior que o ano da venda.')
                is_valid = False
            if yrrenovated > sale_date.year:
                st.warning('O ano da reforma (yrrenovated) não pode ser maior que o ano da venda.')
                is_valid = False
            if yrrenovated > 0 and yrrenovated < yrbuilt:
                st.warning('O ano da reforma (yrrenovated) não pode ser menor que o ano de construção.')
                is_valid = False
            if (sale_date.year - yrbuilt) < 0:
                st.warning('A idade do imóvel (property_age) não pode ser negativa.')
                is_valid = False
            if yrrenovated > 0 and (sale_date.year - yrrenovated) < 0:
                st.warning('O tempo desde a reforma (years_since_renovation) não pode ser negativo.')
                is_valid = False

            if is_valid:
                # Calculo automatico das variaveis derivadas
                sale_year = sale_date.year
                sale_month = sale_date.month
                sale_quarter = ((sale_month - 1) // 3) + 1

                property_age = sale_year - yrbuilt

                was_renovated = int(yrrenovated > 0)

                years_since_renovation = (
                    sale_year - yrrenovated
                    if yrrenovated > 0
                    else None
                )

                has_basement = int(sqft_basement > 0)
                living_to_lot_ratio = sqft_living / sqft_lot
                bathrooms_per_bedroom = bathrooms / bedrooms

                new_property = pd.DataFrame([{
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'sqft_living': sqft_living,
                    'sqft_lot': sqft_lot,
                    'floors': floors,
                    'waterfront': waterfront,
                    'view': view,
                    'condition': condition,
                    'grade': grade,
                    'sqft_above': sqft_above,
                    'sqft_basement': sqft_basement,
                    'zipcode': zipcode,
                    'lat': latitude,
                    'long': longitude,
                    'sale_year': sale_year,
                    'sale_month': sale_month,
                    'sale_quarter': sale_quarter,
                    'property_age': property_age,
                    'was_renovated': was_renovated,
                    'years_since_renovation': years_since_renovation,
                    'has_basement': has_basement,
                    'living_to_lot_ratio': living_to_lot_ratio,
                    'bathrooms_per_bedroom': bathrooms_per_bedroom
                }])

                try:
                    # Checagem programatica das colunas esperadas pelo pre-processador
                    if hasattr(house_price_model, 'named_steps') and 'preprocessor' in house_price_model.named_steps:
                        preproc = house_price_model.named_steps['preprocessor']
                        if hasattr(preproc, 'feature_names_in_'):
                            expected_columns = list(preproc.feature_names_in_)
                        else:
                            expected_columns = list(new_property.columns)
                    else:
                        expected_columns = list(new_property.columns)

                    missing_columns = set(expected_columns) - set(new_property.columns)

                    if missing_columns:
                        st.error(
                            f'Não foi possível realizar a previsão. Colunas ausentes: '
                            f'{sorted(missing_columns)}'
                        )
                    else:
                        new_property = new_property[expected_columns]
                        predicted_price = house_price_model.predict(new_property)[0]

                        st.success('Previsão gerada com sucesso.')

                        st.metric(
                            'Preço estimado de venda',
                            f'${predicted_price:,.2f}'
                        )

                        

                        with st.expander('Ver dados enviados ao modelo'):
                            st.dataframe(new_property, use_container_width=True)

                except Exception as e:
                    st.error(f'Erro ao realizar a previsão: {e}')



