import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

os.putenv("TF_CPP_MIN_LOG_LEVEL", "2")
os.putenv("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import pandas as pd
import seaborn as sns
import tsfel
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC, SVR
from tqdm import tqdm
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def depression_classifier_data(control_data, condition_data):
    depression_classifier = {"X": [], "y": []}
    for data in condition_data.values():
        sleep_data = data["sleep-data"]
        # get by sleep vectors by individual dates
        unique_dates = set([x["date"] for x in sleep_data])
        for date in unique_dates:
            activities = []
            for x in sleep_data:
                if x["date"] == date:
                    activity = x["activity"]
                    activities.append(activity)
            if len(activities) == 1440:
                depression_classifier["X"].append(activities)
                depression_classifier["y"].append(1)  # 1 for depression
    for data in control_data.values():
        sleep_data = data["sleep-data"]
        # get by sleep vectors by individual dates
        unique_dates = set([x["date"] for x in sleep_data])
        for date in unique_dates:
            activities = []
            for x in sleep_data:
                if x["date"] == date:
                    activity = x["activity"]
                    activities.append(activity)
            if len(activities) == 1440:
                depression_classifier["X"].append(activities)
                depression_classifier["y"].append(0)  # 0 for non-depression
    return depression_classifier


# Load data
def load_data():
    with open("control.json", "r") as f_control:
        control_data = json.load(f_control)
    with open("condition.json", "r") as f_condition:
        condition_data = json.load(f_condition)
    depression_classifier_data_dict = depression_classifier_data(
        control_data, condition_data
    )
    return depression_classifier_data_dict


def build_random_forest_model(X, y):
    print("Building Random Forest Model")
    model = RandomForestClassifier()
    model.fit(X, y)
    kfold = KFold(n_splits=10)
    y_pred = cross_val_predict(model, X, y, cv=kfold)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print(
        f"Random Forest Model Evaluation:\n Accuracy: {accuracy}\n F1: {f1}\n Recall: {recall}\n Feature Importances: {model.feature_importances_}"
    )
    print(model.feature_importances_)
    return model


def build_xgboost_model(X, y):
    print("Building XGBoost Model")
    model = XGBClassifier()
    model.fit(X, y)
    kfold = KFold(n_splits=10)
    y_pred = cross_val_predict(model, X, y, cv=kfold)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print(
        f"XGBoost Model Evaluation:\n Accuracy: {accuracy}\n F1: {f1}\n Recall: {recall}\n Feature Importances: {model.feature_importances_}"
    )
    return model


def svm_time_series_model(X, y):
    try:
        # Time series SVC
        print("Building Time Series SVC Model")
        # Scale data
        X = TimeSeriesScalerMinMax().fit_transform(X)
        model = TimeSeriesSVC(
            kernel="gak",
            C=10.0,
            gamma=0.1,
            verbose=1,
            probability=True,
            random_state=42,
        )
        model.fit(X, y)
        return model
    except Exception as e:
        print(e)
        return None


def tsfel_model(X, y):
    # Extract features
    try:
        print("Extracting Features")
        features = tsfel.time_series_features_extractor(X, fs=1, verbose=0)
        model = build_random_forest_model(features, y)
        return model
    except Exception as e:
        print(e)
        return None


def RNN_model(X_train, y_train, X_val, y_val):
    try:
        print("Building RNN Model")
        model = Sequential()
        model.add(LSTM(256, input_shape=(1, X_train.shape[1]), return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(1, activation="relu"))
        model.summary()
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_train.reshape(-1, 1, X_train.shape[1]),
            y_train,
            epochs=50,
            batch_size=128,
            validation_data=(X_val.reshape(-1, 1, X_val.shape[1]), y_val),
        )
        # plot accuracy and loss epochs
        plt.plot(model.history.history["accuracy"])
        plt.plot(model.history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.show()
        return model
    except Exception as e:
        print(e)
        return None


def evaluate_model(model, X_test, y_test, model_name="Random Forest"):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        with open(f"results/{model_name}.txt", "w+") as f:
            f.write(f"Test Accuracy: {accuracy}\n")
            f.write(f"Test F1: {f1}\n")
            f.write(f"Test Recall: {recall}\n")
            f.write("Confusion Matrix\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
        print(f"{model_name} Test Accuracy: {accuracy}")
        return model
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    # Load data
    depression_classifier_data_dict = load_data()
    X = np.array(depression_classifier_data_dict["X"]).astype(np.float32)
    y = np.array(depression_classifier_data_dict["y"]).astype(np.float32)
    # convert to dataframe
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y)

    # Split data
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    print("Data Split: Train Size: ", X_train.shape, " Test Size: ", X_test.shape)

    # Train and evaluate models
    models = []
    evaluation_results = []
    # models.append(build_random_forest_model(X_train, y_train))
    # evaluation_results.append(
    #     evaluate_model(models[-1], X_test, y_test, "Random Forest")
    # )
    # models.append(build_xgboost_model(X_train, y_train))
    # evaluation_results.append(evaluate_model(models[-1], X_test, y_test, "XGBoost"))
    # models.append(tsfel_model(X_train, y_train))
    # evaluation_results.append(evaluate_model(models[-1], X_test, y_test, "TSFEL"))
    # models.append(RNN_model(X_train, y_train, X_val, y_val))
    models.append(svm_time_series_model(X_train, y_train))
    evaluation_results.append(evaluate_model(models[-1], X_test, y_test, "SVM"))
