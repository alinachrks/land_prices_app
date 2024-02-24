import streamlit as st
from streamlit_option_menu import option_menu
from components import land_prediction, strategy, estimation, methodology, about, crime, life_quality


# Apply theme from the config file
st.set_page_config(
    page_title="Аналитика",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # Create a sidebar option menu
        with st.sidebar:
            app = option_menu(
                menu_title='📈Аналитика',
                options=['🏠 Прогноз стоимости','🏡 Оценка привлекательности','🎯 Расчет по стратегии', '👮🏻‍♂️ Преступность', '🌱 Качество жизни', '📒 Методика','📃 О приложении'],
                icons=['house-garden','house-garden','house-garden','house-garden','house-garden', 'house-garden', 'house-garden'],
                menu_icon='house-garden',
                default_index=0,  # Change the default index to 0 for "🏠 Прогноз стоимости"
                styles={
                    "container": {"padding": "5!important", "width": "100%"},  # Adjust width here
                    # "icon": {"color": "white", "font-size": "0px"},
                    "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "orange"},
                    "nav-link-selected": {"background-color": "#44484d"},
                }
            )

        # Display selected app based on user choice
        if app == "🏠 Прогноз стоимости":
            land_prediction.app()
        elif app == "🏡 Оценка привлекательности":
            estimation.app()    
        elif app == "🎯 Расчет по стратегии":
            strategy.app()        
        elif app == '👮🏻‍♂️ Преступность':
            crime.app()
        elif app == '🌱 Качество жизни':
            life_quality.app()   
        elif app == '📒 Методика':
            methodology.app()
        elif app == '📃 О приложении':
            about.app()
         

# Create an instance of MultiApp and add your apps
multi_app = MultiApp()

# Add your apps to the MultiApp instance
multi_app.add_app("🏠 Прогноз стоимости", land_prediction.app)
multi_app.add_app("🏡 Оценка привлекательности", estimation.app)
multi_app.add_app("🎯 Расчет по стратегии", strategy.app)
multi_app.add_app("👮🏻‍♂️ Преступность", crime.app)
multi_app.add_app("🌱 Качество жизни", life_quality.app)
multi_app.add_app("📒 Методика", methodology.app)
multi_app.add_app("📃 О приложении", about.app)

# Run the MultiApp
multi_app.run()
