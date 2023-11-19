from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, precision_score, recall_score, roc_auc_score,
                            roc_curve, f1_score)
from pickle import dump, load
import pandas as pd
import streamlit as st



def split_data(df: pd.DataFrame):
    y = df['MetabolicSyndrome']
    X = df[["Age", "WaistCirc", "BMI", "Albuminuria", "UrAlbCr", "UricAcid",
            "BloodGlucose", "HDL", "Triglycerides"]]

    return X, y


def open_data(path="data/data.csv"):
    df = pd.read_csv(path)
    df = df[['MetabolicSyndrome', "Age", "WaistCirc", "BMI", "Albuminuria", "UrAlbCr", "UricAcid",
            "BloodGlucose", "HDL", "Triglycerides"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    #to_encode = ['Sex']
    #for col in to_encode:
        #dummy = pd.get_dummies(X_df[col], prefix=col)
        #X_df = pd.concat([X_df, dummy], axis=1)
        #X_df.drop(col, axis=1, inplace=True)

    if test:
       return X_df, y_df
    else:
       return X_df


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    tree = DecisionTreeClassifier(random_state=40, max_depth=2, class_weight="balanced")
    model = BaggingClassifier(estimator=tree, n_estimators=100,
                        random_state=40,
                        max_samples=0.3)
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    precision = precision_score(test_prediction, y_df)
    print(f"Model precision is {precision}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Метаболического синдрома нет с вероятностью",
        1: "Метаболический синдром есть с вероятностью"
    }

    encode_prediction = {
        0: "Метаболического синдрома нет",
        1: "Метаболический синдром"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)