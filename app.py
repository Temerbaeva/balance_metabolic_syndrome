import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/metabolic-img.jpeg')

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


def write_user_data(df):
    st.write("## Данные пациента")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные параметры пациента')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    #sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    age = st.sidebar.slider("Возраст", min_value=1, max_value=90, value=0,
                            step=1)

    waistcirc = st.sidebar.slider(
        "Окружность талии",
        min_value=40, max_value=180, value=0, step=1)

    bmi = st.sidebar.slider("Индекс массы тела", min_value=10, max_value=70, value=0,
                            step=1)

    albuminuria = st.sidebar.slider("Уровень альбуминурии",
                               min_value=0, max_value=5, value=0, step=1)

    uralbcr = st.sidebar.slider("Соотношение альбуминурии и креатина в моче",
                                    min_value=0, max_value=6000, value=0, step=1)

    uricacid = st.sidebar.slider("Уровень мочевой кислоты в крови",
                                min_value=0, max_value=12, value=0, step=1)

    bloodglucose = st.sidebar.slider("Уровень глюкозы в крови",
                                 min_value=30, max_value=400, value=0, step=1)

    hdl = st.sidebar.slider("Уровень холестерина липопротеинов высокой плотности",
                                     min_value=10, max_value=160, value=0, step=1)

    triglycerides = st.sidebar.slider("Уровень триглицерида",
                            min_value=20, max_value=1600, value=0, step=1)

    #translatetion = {
       # "Мужской": "Male",
       # "Женский": "Female"}

    data = {
        #"Sex": translatetion[sex],
        "Age": age,
        "WaistCirc": waistcirc,
        "BMI": bmi,
        "Albuminuria": albuminuria,
        "UrAlbCr": uralbcr,
        "UricAcid": uricacid,
        "BloodGlucose": bloodglucose,
        "HDL": hdl,
        "Triglycerides": triglycerides
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()