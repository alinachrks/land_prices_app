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
    
    # Create a Streamlit caching decorator for data loading and model training
    @st.cache_data  # Cache the data and model
    def load_data_and_train_model():
        # Load your CSV data
        data = pd.read_csv('data/train_data.csv')
    
        # Split the dataset into input features (X_train) and target (y_train)
        X_train = data.drop(columns=['id', 'price']).values
        y_train = data['price'].values
    
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
    
        return data, model
    
    # Call the cached function to load data and train model
    data, model = load_data_and_train_model()
    
    # Create the Streamlit app interface
    st.title("Расчет по стратегии")
    
    # Add an image after the title
    st.image("img/pic1.jpeg", use_column_width=True)
    
    @st.cache_data  # Cache the data and model
    def load_land_area():
        # Load your CSV data
        csv_data = pd.read_csv('data/land_area_updated.csv')
        return csv_data
    
    # Load your CSV data
    csv_data = load_land_area()
    
    # Function to format price as an integer (removing extra zeros)
    def format_price(price):
        return f"₸ {int(price):,}"  # Format as an integer
    
    # Create a container for the map
    with st.container():
        # Create a base map
        @st.cache_resource
        def create_map():
            m = folium.Map(location=[43.238293, 76.912471], zoom_start=9, control_scale=True, width=700)
    
            # Create a MarkerCluster for clustering
            marker_cluster = MarkerCluster().add_to(m)
    
            # Add markers for each land plot with popups
            for index, row in csv_data.iterrows():
                formatted_price = format_price(row['price'])  # Format the price using the function
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"<b>Адрес:</b> {row['address']}<br><b>Площадь:</b> {row['area']} sq.m<br><b>Цена:</b> {formatted_price}",
                ).add_to(marker_cluster)  # Add markers to the MarkerCluster for clustering
    
            return m
    
    # Call the create_map function to create or retrieve the cached map
    m = create_map()
    
    # Display the map in Streamlit using HTML with responsive height
    st.header("Карта земельных участков")
    st.components.v1.html(m._repr_html_(), width=710, height=400)
    
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
    
    
    st.markdown("""
    ##### Прогноз стоимости земли после программы реновации
                """)
    
    # Characteristics for each strategy
    characteristics_high_density_population = {
        "distance_1000m_ddo": True,
        "distance_1000m_schools": True,
        "distance_1000m_medical": True,
        "is_parking_exists": True,
        "distance_park_1000m": True,
        "distance_bikeroad_1000m": True,
        "deficit_ddo": 0,
        "deficit_schools": 0,
        "amount_dosug_1000m": 10,
        "amount_of_cameras_1000m": 5,
        "amount_of_bins_1000m": 20,
        "amount_of_poi_1000m": 15,
        "amount_of_business19_1000m": 8,
        "index_of_nearest_sensor_pm": 25,
        "index_of_nearest_sensor_am": 20
    }

    characteristics_current_density_population = {
        "distance_1000m_ddo": True,
        "distance_1000m_schools": True,
        "distance_1000m_medical": True,
        "is_parking_exists": True,
        "distance_park_1000m": True,
        "distance_bikeroad_1000m": True,
        "deficit_ddo": 1,
        "deficit_schools": 1,
        "amount_dosug_1000m": 8,
        "amount_of_cameras_1000m": 4,
        "amount_of_bins_1000m": 15,
        "amount_of_poi_1000m": 10,
        "amount_of_business19_1000m": 5,
        "index_of_nearest_sensor_pm": 30,
        "index_of_nearest_sensor_am": 25
    }

    characteristics_sustainable_development = {
        "distance_1000m_ddo": True,
        "distance_1000m_schools": True,
        "distance_1000m_medical": True,
        "is_parking_exists": True,
        "distance_park_1000m": True,
        "distance_bikeroad_1000m": True,
        "deficit_ddo": 0,
        "deficit_schools": 0,
        "amount_dosug_1000m": 12,
        "amount_of_cameras_1000m": 6,
        "amount_of_bins_1000m": 25,
        "amount_of_poi_1000m": 20,
        "amount_of_business19_1000m": 10,
        "index_of_nearest_sensor_pm": 20,
        "index_of_nearest_sensor_am": 15
    }

    # Dictionary to map each strategy to its characteristics
    strategy_characteristics = {
        "Высокая плотность населения": characteristics_high_density_population,
        "Текущая плотность населения": characteristics_current_density_population,
        "Устойчивое развитие": characteristics_sustainable_development
    }

    property_characteristics = [
        "Участок, прилегающий к Роще Баума, в квадрате улиц Сейфуллина, Акан Серы, Успенского, Стоимость, тенге/м² 73 000",
        "Участок на квадрате улиц Кассина, Акан Сери, Сейфуллина, Хетагурова, Стоимость, тенге/м² 75 000",
        "Участок в квадрате улиц Сейфуллина, Котельникова, Акан Серы, Стоимость, тенге/м² 70 000",
        "Участок в квадрате улиц Ауэзова, Жандосова, Манаса, М.Озтюрка, Стоимость, тенге/м² 232 000",
        "Участок в квадрате улиц Манаса, Бухар жырау, Ауэзова, Габдуллина, Стоимость, тенге/м² 231 000",
        "Участок на пересечении улиц Бенберина - Тополиная и Бенберина - Шугыла в мкр.Айгерим-1, Стоимость, тенге/м² 56 000",
        "Участок на пересечении улиц Кисловодская-Левского и 2-ая Кисловодская-Отрарская, Стоимость, тенге/м² 72 000",
        "Участок в квадрате улиц Абая, Тургут Озала, Брусиловского, Толе би, Стоимость, тенге/м² 82 000",
        "Участок в квадрате улиц Розыбакиева, Толе би, И.Каримова, Карасай батыра, Стоимость, тенге/м² 90 000",
        "Участок в мкр. Мамыр - западнее улицы Яссауи, Стоимость, тенге/м² 108 000"
    ]


    # Create a dropdown to select property characteristics
    selected_property = st.selectbox("Выберите участок", property_characteristics)

    # Extract the property and price
    property_info, price_info = selected_property.rsplit("Стоимость, тенге/м²", 1)
    property_info = property_info.strip()
    price_info = price_info.strip()


    # Display the selected property with increased font size and bold price
    st.write("Вы выбрали следующий участок:")
    st.write(property_info)
    st.markdown(f"**Стоимость, тенге/м {price_info}**")


    submenu = ["Высокая плотность населения", "Текущая плотность населения", "Устойчивое развитие"]
    selected_strategy = st.selectbox("Выбор стратегии", submenu)

    # Retrieve the selected strategy
    selected_characteristics = strategy_characteristics[selected_strategy]

    # Input fields for property characteristics
    st.header("Характеристики земельного участка")

    # Define the prescriptive message for each strategy
    prescriptive_messages = {
        "Высокая плотность населения": """
            <div style="background-color:#e4e4e4;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
                <h3 style="text-align:justify;color:black;padding:10px">Высокая плотность населения</h3>
                <ul>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Плотность населения:</b> Более 11 450 чел./км²
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Паркинг:</b> Более 1900 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Кол-во мест в ДДО (дошкольные детские организации):</b> Более 800 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Кол-во мест в школе:</b> Более 4800 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Площадь озеленения:</b> Менее 0.2 га
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Количество фонарей:</b> Около 80 штук
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Количество камер:</b> Около 30 штук
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Коммерческих помещений:</b> Около 300 м²
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Велодорожка:</b> 1000 м
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Комментарии:</b> Этот сценарий предполагает увеличение плотности населения до уровней, которые могут привести к перегрузке инфраструктуры и общественных услуг. В то время как застройщики могут извлечь выгоду из более высокой плотности зданий и возможности увеличения доходов от сдачи в аренду, такие условия часто сопровождаются ухудшением качества жизни. Проблемы с парковкой, перенаселенностью школ, недостаточное количество зеленых зон и потенциально недостаточное количество фонарей и камер для обеспечения безопасности могут создать нежелательные социальные и экологические условия.
                    </li>
                </ul>
            </div>
        """,
        "Текущая плотность населения": """
            <div style="background-color:#e4e4e4;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
                <h3 style="text-align:justify;color:black;padding:10px">Текущая плотность населения</h3>
                <ul>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Плотность населения:</b> Менее 11 450 чел./км²
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Паркинг:</b> Около 2360 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Кол-во мест в ДДО:</b> До 300 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Кол-во мест в школе:</b> До 3600 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Площадь озеленения:</b> Более 0.4 га
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Количество фонарей:</b> Около 150 штук
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Количество камер:</b> Около 60 штук
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Коммерческих помещений:</b> Более 1000 м²
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Велодорожка:</b> Около 2500 м
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Комментарии:</b> Этот сценарий подразумевает поддержание текущего уровня плотности населения, который предоставляет достаточное количество мест для парковки и более широкие возможности для озеленения. Относительное увеличение числа фонарей и камер может способствовать улучшению общественной безопасности. Похоже, что текущая плотность предлагает более сбалансированный подход к развитию городского пространства, позволяя избежать экстремальной перенаселенности и создавая условия для более комфортной городской жизни.
                    </li>
                </ul>
            </div>
        """,
        "Устойчивое развитие": """
            <div style="background-color:#e4e4e4;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
                <h3 style="text-align:justify;color:black;padding:10px">Устойчивое развитие</h3>
                <ul>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Плотность населения:</b> 11 450 чел./км²
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Паркинг:</b> Около 1670 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Кол-во мест в ДДО:</b> Около 600 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Кол-во мест в школе:</b> Около 4800 мест
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Площадь озеленения:</b> Около 0.3 га
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Количество фонарей:</b> Около 120 штук
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Количество камер:</b> Около 45 штук
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Коммерческих помещений:</b> Около 900 м²
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Велодорожка:</b> Около 1800 м
                    </li>
                    <li style="text-align:justify;color:black;padding:10px">
                        <b>Комментарии:</b> Этот сценарий является предпочтительной стратегией, поскольку оно ориентировано на баланс между ростом и качеством жизни. Предложения по устойчивому развитию указывают на умеренное количество парковочных мест и школьных мест, сочетая это с адекватным количеством озеленения. Увеличение количества фонарей и камер обеспечит лучшую освещенность и безопасность, а подходящее количество коммерческих помещений и развитая сеть велодорожек будут способствовать экономическому росту и здоровому образу жизни. Этот сценарий подразумевает осознанный подход к развитию, который учитывает долгосрочные последствия для городской среды и благополучия его жителей.
                    </li>
                </ul>
            </div>
        """
    }


    # Display the prescriptive message based on the selected strategy
    if selected_strategy in prescriptive_messages:
        st.markdown(prescriptive_messages[selected_strategy], unsafe_allow_html=True)
    else:
        st.error("Выберите стратегию для отображения рекомендаций.")

    with st.form(key="input_form"):
        # Populate input fields based on the selected strategy
        # Checkbox inputs with custom CSS for disabled state
        # Number input fields with custom CSS for disabled state
        # deficit_ddo = st.number_input("Дефицит ДДО", min_value=0, value=selected_characteristics["deficit_ddo"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )

        # deficit_schools = st.number_input("Дефицит школ", min_value=0, value=selected_characteristics["deficit_schools"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )

        # amount_dosug_1000m = st.number_input("Количество объектов досуга в радиусе 1000м", min_value=0, value=selected_characteristics["amount_dosug_1000m"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )

        # amount_of_cameras_1000m = st.number_input("Количество камер видеонаблюдения в радиусе 1000м", min_value=0, value=selected_characteristics["amount_of_cameras_1000m"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )

        # amount_of_bins_1000m = st.number_input("Количество коммерческих организации в радиусе 1000м", min_value=0, value=selected_characteristics["amount_of_bins_1000m"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )

        # amount_of_poi_1000m = st.number_input("Количество объектов благоустройства в радиусе 1000м", min_value=0, value=selected_characteristics["amount_of_poi_1000m"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )

        # amount_of_business19_1000m = st.number_input("Количество предприятий общественного питания в радиусе 1000м", min_value=0, value=selected_characteristics["amount_of_business19_1000m"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )

        # index_of_nearest_sensor_pm = st.number_input("Показатель датчика качества воздуха после полудня", min_value=0, value=selected_characteristics["index_of_nearest_sensor_pm"], disabled=True)
        # st.markdown(
        #     "<style>input[type='number'][disabled] {background-color: inherit;}</style>", 
        #     unsafe_allow_html=True
        # )
        submitted = st.form_submit_button("Предсказать стоимость")

    
    # Button to make predictions
    if submitted:
        # Data preprocessing
        # features = np.array([
        #                 deficit_ddo, deficit_schools, amount_dosug_1000m,
        #                 amount_of_cameras_1000m, amount_of_bins_1000m,
        #                 amount_of_poi_1000m, amount_of_business19_1000m, index_of_nearest_sensor_pm,
        #                 ]).reshape(1, -1)
        
        # # Predict property price
        # predicted_price = model.predict(features)[0]
        
        # Apply the minimum and maximum limits
        # predicted_price = min(max(predicted_price, 80000), 350000)

        # Удаление пробелов из строки цены и преобразование ее в целое число
        # Удаление всех символов, кроме цифр, из строки цены и преобразование ее в число с плавающей точкой
        price_info = ''.join(filter(str.isdigit, price_info))
        price_numeric = float(price_info)

        # Вычисляем предсказанную цену, умножая текущую цену на коэффициент
        # Словарь с коэффициентами для каждой функции
        coefficients = {
            "Высокая плотность населения": 1.275,
            "Текущая плотность населения": 1.5,
            "Устойчивое развитие": 1.8
        }

        # Получение коэффициента для выбранной стратегии
        coefficient = coefficients[selected_strategy]

        # Вычисление предсказанной цены с использованием коэффициента
        predicted_price = np.round((price_numeric * coefficient), 0)


        # Display the prediction result
        st.header("Результат прогноза")
        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] {
                font-size: 40px;
                color: green;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.metric(label='Стоимость земли (м²)', value=f"₸ {predicted_price:.0f}", delta=None)
    