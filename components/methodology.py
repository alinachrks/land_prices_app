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
    st.title("Методика")

    # Add an image after the title
    st.image("img/pic1.jpeg", use_column_width=True)

    # Description
    text = """
    Для повышения эффективности городского планирования была проделана следующая работа:

    1. Применена модель машинного обучения для обработки и анализа больших объемов данных.
    2. Проведен социально-демографический анализ, использующий данные от различных организаций и учреждений города Алматы.
    3. Проведен анализ цен на недвижимость и земельные участки, с использованием данных с ресурсов, таких как krisha.kz.
    4. Проведен анализ нагрузки на улично-дорожную сеть с использованием данных Google и 2GIS.
    5. Проведен анализ компаний, зарегистрированных в городе Алматы на основе данных Taldau.
    """

    st.markdown(text)

    st.header("Анализ данных")

    # Add an image after the title
    st.image("img/meth_1.jpeg", use_column_width=True)

    st.header("Машинное обучение")

    # Add an image after the title
    st.image("img/meth_2.jpeg", use_column_width=True)



