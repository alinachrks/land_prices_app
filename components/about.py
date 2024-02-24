import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster  # Import MarkerCluster for clustering
import plotly.express as px

def app():

    # Create the Streamlit app interface
    st.title("О приложении")

    # Add an image after the title
    st.image("img/about.png", use_column_width=True)

    # Description
    st.markdown("""
    ##### Веб-приложение для прогнозирования стоимости земельных участков в Алматы

    Это приложение использует машинное обучение для прогнозирования стоимости земельных участков. 
    Оно загружает предварительно обученную модель линейной регрессии, которая принимает на вход различные характеристики участка, 
    такие как расстояние до ДДО, школы, медучреждения, дефицит ДДО и школ, количество объектов досуга, наличие парковки, парка, камер видеонаблюдения, 
    наличие велодорожки, количество мусорных контейнеров, количество точек интереса и предприятий общественного питания в радиусе 1000 метров, 
    индекс ближайшего датчика после полудня и до полудня. 
    Приложение предварительно обрабатывает входные данные, объединяя некоторые характеристики и добавляя новые, например, расстояние до ближайшего города.
    """)





