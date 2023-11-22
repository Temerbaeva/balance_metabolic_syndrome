import pandas as pd     ### импортируем библиотеки
import streamlit as st   
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict   ###обученная модель


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():                                ###интерфейс
    image = Image.open('data/metabolic-img.jpeg')    ###загрузка датасета

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Metabolic syndrome",
        page_icon=image,
    )

    st.write(
        """
        # Метаболический синдром 
        На основании данных пациентов определяем метаболический синдром.
        """
    )

    st.image(image)


def write_user_data(df):            ###передадим как аргумент данные пациента
    st.write("## Данные пациента")
    st.write(df)


def write_prediction(prediction, prediction_probas):    ###функция передачи аргументов прогноза
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():             ###оформляем ввод данных и их обработку
    st.sidebar.header('Заданные параметры пациента')    ###заголовок боковой панели
    user_input_df = sidebar_input_features()            ###передадим в переменную признаки пользователя

    train_df = open_data()                              ###передадим датасет в переменную
    train_X_df, _ = split_data(train_df)                ###выделим признаки
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)    ###добавим к признакам данные пользователя
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)               ###выведем как аргументы данные пользователя

    prediction, prediction_probas = load_model_and_predict(user_X_df)   ###в сохраненную обученную модель передаем данные пользователя
    write_prediction(prediction, prediction_probas)   


def sidebar_input_features():      ###ввод признаков пользователя в боковой панели
    
    age = st.sidebar.slider("Возраст", min_value=1, max_value=90, value=0,   ###ввод данных слайдером
                            step=1)

    waistcirc = st.sidebar.slider(
        "Окружность талии (см)",
        min_value=40, max_value=180, value=0, step=1)

    bmi = st.sidebar.slider("Индекс массы тела", min_value=10, max_value=70, value=0,
                            step=1)

    bloodglucose = st.sidebar.slider("Уровень глюкозы в крови (мг/дл)",
                                 min_value=30, max_value=400, value=0, step=1)

    hdl = st.sidebar.slider("Уровень холестерина липопротеинов высокой плотности (мг/дл)",
                                     min_value=10, max_value=160, value=0, step=1)

    triglycerides = st.sidebar.slider("Уровень триглицерида (мг/дл)",
                            min_value=20, max_value=1600, value=0, step=1)

    data = {
        "Age": age,
        "WaistCirc": waistcirc,
        "BMI": bmi,
        "BloodGlucose": bloodglucose,
        "HDL": hdl,
        "Triglycerides": triglycerides
    }

    df = pd.DataFrame(data, index=[0])       ###Введенные пользователем данныесохраним в датасет

    return df


if __name__ == "__main__":          ###определим точку входа
    process_main_page()
