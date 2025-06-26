# import yfinance as yf
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_score
# import matplotlib.pyplot as plt
# import io
# import base64
# import pytz  
  
# def load_data(ticker, interval="1mo", start_date=None, end_date=None):
#     # Use yfinance start/end params if dates are provided
#     if start_date or end_date:
#         print(f"Loading data for {ticker} from {start_date} to {end_date} with interval {interval}")
#         data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
#     else:
#         print(f"Loading 10 years data for {ticker} with interval {interval}")
#         data = yf.Ticker(ticker).history(period="10y", interval=interval)
    
#     print(f"Fetched {len(data)} rows for {ticker}")
#     if data.empty:
#         raise ValueError("No data found for this ticker.")
#     data.index = pd.to_datetime(data.index)
#     return data

# def engineer_features(data):
#     data["Tomorrow"] = data["Close"].shift(-1)
#     data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
#     data = data.copy()

#     available_rows = data.shape[0]
#     horizons = [h for h in [2, 5, 60, 250, 1000] if h < available_rows // 2]
#     new_predictors = []

#     for horizon in horizons:
#         rolling_averages = data.rolling(horizon).mean()
#         ratio_col = f"Close_Ratio_{horizon}"
#         trend_col = f"Trend_{horizon}"

#         data[ratio_col] = data["Close"] / rolling_averages["Close"]
#         data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]

#         new_predictors += [ratio_col, trend_col]

#     data.dropna(subset=[col for col in data.columns if col != "Tomorrow"], inplace=True)
#     return data, new_predictors

# def predict(train, test, predictors, model):
#     model.fit(train[predictors], train["Target"])
#     preds = model.predict_proba(test[predictors])[:, 1]
#     preds = (preds >= 0.6).astype(int)
#     return pd.DataFrame({
#         "Target": test["Target"],
#         "Predictions": preds
#     }, index=test.index)

# def backtest(data, model, predictors, start=None, step=250):
#     all_predictions = []

#     if not start or start >= data.shape[0]:
#         start = max(20, int(data.shape[0] * 0.2))

#     for i in range(start, data.shape[0], step):
#         train = data.iloc[0:i].copy()
#         test = data.iloc[i:(i + step)].copy()

#         # Relaxed sizes for debugging
#         if test.empty or train.empty or len(test) < 5 or len(train) < 20:
#             continue

#         predictions = predict(train, test, predictors, model)
#         all_predictions.append(predictions)

#     if not all_predictions:
#         raise ValueError("No valid prediction windows found. Try using a stock with more historical data.")

#     return pd.concat(all_predictions)

# def plot_actual(data, graph_style="line"):
#     fig, ax = plt.subplots(figsize=(18, 6))
#     if graph_style == "line":
#         data["Target"].plot(ax=ax, color="blue", label="Actual")
#     elif graph_style == "bar":
#         ax.bar(data.index, data["Target"], color="blue", label="Actual")
#     elif graph_style == "step":
#         ax.step(data.index, data["Target"], color="blue", label="Actual")
#     elif graph_style == "scatter":
#         ax.scatter(data.index, data["Target"], color="blue", label="Actual", s=10)
#     else:
#         data["Target"].plot(ax=ax, color="blue", label="Actual")  # fallback

#     ax.legend()
#     ax.set_title("Actual Target")
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = base64.b64encode(buf.read()).decode("utf-8")
#     buf.close()
#     plt.close(fig)
#     return img


# def plot_predicted(data, graph_style="line"):
#     fig, ax = plt.subplots(figsize=(18, 6))
#     if graph_style == "line":
#         data["Predictions"].plot(ax=ax, color="orange", label="Predicted")
#     elif graph_style == "bar":
#         ax.bar(data.index, data["Predictions"], color="orange", label="Predicted")
#     elif graph_style == "step":
#         ax.step(data.index, data["Predictions"], color="orange", label="Predicted")
#     elif graph_style == "scatter":
#         ax.scatter(data.index, data["Predictions"], color="orange", label="Predicted", s=10)
#     else:
#         data["Predictions"].plot(ax=ax, color="orange", label="Predicted")  # fallback

#     ax.legend()
#     ax.set_title("Predicted Target")
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = base64.b64encode(buf.read()).decode("utf-8")
#     buf.close()
#     plt.close(fig)
#     return img

# def plot_pie_chart(distribution, title="Distribution"):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     labels = [f"{k}: {v}" for k, v in distribution.items()]
#     values = list(distribution.values())

#     ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
#     ax.set_title(title)
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = base64.b64encode(buf.read()).decode("utf-8")
#     buf.close()
#     plt.close(fig)
#     return img

# def run_model(ticker, interval="1d", start_date=None, end_date=None, graph_style="line"):
#     try:
#         data = load_data(ticker, interval=interval, start_date=start_date, end_date=end_date)
#         data, predictors = engineer_features(data)

#         print(f"Data after feature engineering: {len(data)} rows")

#         model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
#         predictions = backtest(data, model, predictors)

#         precision = precision_score(predictions["Target"], predictions["Predictions"])
#         prediction_distribution = predictions["Predictions"].value_counts().to_dict()
#         target_distribution = (predictions["Target"].value_counts() / predictions.shape[0]).to_dict()

#         img_actual = plot_actual(predictions, graph_style=graph_style)
#         img_predicted = plot_predicted(predictions, graph_style=graph_style)
#         img_target_pie = plot_pie_chart(target_distribution, title="Actual Target Distribution")
#         img_prediction_pie = plot_pie_chart(prediction_distribution, title="Prediction Distribution")

#         return {
#             "ticker": ticker.upper(),
#             "precision": precision,
#             "prediction_distribution": prediction_distribution,
#             "target_distribution": target_distribution,
#             "image_actual": img_actual,
#             "image_predicted": img_predicted,
#             "image_target_pie": img_target_pie,
#             "image_prediction_pie": img_prediction_pie,
#             "status": "success"
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_score
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import io
# import base64
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# def load_data(ticker, interval="1mo", start_date=None, end_date=None):
#     if start_date or end_date:
#         print(f"Loading data for {ticker} from {start_date} to {end_date} with interval {interval}")
#         data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
#     else:
#         print(f"Loading 10 years data for {ticker} with interval {interval}")
#         data = yf.Ticker(ticker).history(period="10y", interval=interval)
    
#     print(f"Fetched {len(data)} rows for {ticker}")
#     if data.empty:
#         raise ValueError("No data found for this ticker.")
#     data.index = pd.to_datetime(data.index)
#     return data

# def engineer_features(data, model_type="randomforest"):
#     data = data.copy()
#     data["Tomorrow"] = data["Close"].shift(-1)
#     data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

#     if model_type == "randomforest":
#         available_rows = data.shape[0]
#         horizons = [h for h in [2, 5, 60, 250, 1000] if h < available_rows // 2]
#         new_predictors = []

#         for horizon in horizons:
#             rolling_averages = data.rolling(horizon).mean()
#             ratio_col = f"Close_Ratio_{horizon}"
#             trend_col = f"Trend_{horizon}"
#             data[ratio_col] = data["Close"] / rolling_averages["Close"]
#             data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]
#             new_predictors += [ratio_col, trend_col]
        
#         data.dropna(subset=[col for col in data.columns if col != "Tomorrow"], inplace=True)
#         return data, new_predictors
#     else:  # LSTM or CNN
#         predictors = ["Close", "Volume", "Open", "High", "Low"]
#         scaler = MinMaxScaler()
#         data[predictors] = scaler.fit_transform(data[predictors])
#         data.dropna(subset=[col for col in data.columns if col != "Tomorrow"], inplace=True)
#         return data, predictors

# def create_sequences(data, predictors, seq_length=60):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[predictors].iloc[i:(i + seq_length)].values)
#         y.append(data["Target"].iloc[i + seq_length])
#     return np.array(X), np.array(y)

# def predict_randomforest(train, test, predictors, model):
#     model.fit(train[predictors], train["Target"])
#     preds = model.predict_proba(test[predictors])[:, 1]
#     preds = (preds >= 0.6).astype(int)
#     return pd.DataFrame({
#         "Target": test["Target"],
#         "Predictions": preds
#     }, index=test.index)

# def predict_neural(train, test, predictors, model, seq_length=60):
#     X_train, y_train = create_sequences(train, predictors, seq_length)
#     X_test, y_test = create_sequences(test, predictors, seq_length)
    
#     if X_train.shape[0] == 0 or X_test.shape[0] == 0:
#         raise ValueError("Not enough data to create sequences.")
    
#     model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
#     preds = model.predict(X_test, verbose=0)
#     preds = (preds >= 0.5).astype(int).flatten()
#     return pd.DataFrame({
#         "Target": y_test,
#         "Predictions": preds
#     }, index=test.index[-len(y_test):])

# def backtest(data, model, predictors, model_type="randomforest", start=None, step=250, seq_length=60):
#     all_predictions = []

#     if not start or start >= data.shape[0]:
#         start = max(20, int(data.shape[0] * 0.2))

#     for i in range(start, data.shape[0], step):
#         train = data.iloc[0:i].copy()
#         test = data.iloc[i:(i + step)].copy()

#         if test.empty or train.empty or len(test) < (seq_length + 5 if model_type in ["lstm", "cnn"] else 5) or len(train) < (seq_length + 20 if model_type in ["lstm", "cnn"] else 20):
#             continue

#         if model_type == "randomforest":
#             predictions = predict_randomforest(train, test, predictors, model)
#         else:
#             predictions = predict_neural(train, test, predictors, model, seq_length)
#         all_predictions.append(predictions)

#     if not all_predictions:
#         raise ValueError("No valid prediction windows found. Try using a stock with more historical data.")

#     return pd.concat(all_predictions)

# def plot_actual(data, graph_style="line"):
#     fig, ax = plt.subplots(figsize=(18, 6))
#     if graph_style == "line":
#         data["Target"].plot(ax=ax, color="blue", label="Actual")
#     elif graph_style == "bar":
#         ax.bar(data.index, data["Target"], color="blue", label="Actual")
#     elif graph_style == "step":
#         ax.step(data.index, data["Target"], color="blue", label="Actual")
#     elif graph_style == "scatter":
#         ax.scatter(data.index, data["Target"], color="blue", label="Actual", s=10)
#     else:
#         data["Target"].plot(ax=ax, color="blue", label="Actual")

#     ax.legend()
#     ax.set_title("Actual Target")
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = base64.b64encode(buf.read()).decode("utf-8")
#     buf.close()
#     plt.close(fig)
#     return img

# def plot_predicted(data, graph_style="line"):
#     fig, ax = plt.subplots(figsize=(18, 6))
#     if graph_style == "line":
#         data["Predictions"].plot(ax=ax, color="orange", label="Predicted")
#     elif graph_style == "bar":
#         ax.bar(data.index, data["Predictions"], color="orange", label="Predicted")
#     elif graph_style == "step":
#         ax.step(data.index, data["Predictions"], color="orange", label="Predicted")
#     elif graph_style == "scatter":
#         ax.scatter(data.index, data["Predictions"], color="orange", label="Predicted", s=10)
#     else:
#         data["Predictions"].plot(ax=ax, color="orange", label="Predicted")

#     ax.legend()
#     ax.set_title("Predicted Target")
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = base64.b64encode(buf.read()).decode("utf-8")
#     buf.close()
#     plt.close(fig)
#     return img

# def plot_pie_chart(distribution, title="Distribution"):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     labels = [f"{k}: {v}" for k, v in distribution.items()]
#     values = list(distribution.values())

#     ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
#     ax.set_title(title)
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = base64.b64encode(buf.read()).decode("utf-8")
#     buf.close()
#     plt.close(fig)
#     return img

# def run_model(ticker, interval="1d", start_date=None, end_date=None, graph_style="line", model="randomforest"):
#     try:
#         data = load_data(ticker, interval=interval, start_date=start_date, end_date=end_date)
#         data, predictors = engineer_features(data, model_type=model)

#         print(f"Data after feature engineering: {len(data)} rows")

#         if model == "randomforest":
#             model_instance = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
#         elif model == "lstm":
#             model_instance = Sequential([
#                 LSTM(50, return_sequences=True, input_shape=(60, len(predictors))),
#                 Dropout(0.2),
#                 LSTM(50),
#                 Dropout(0.2),
#                 Dense(25),
#                 Dense(1, activation="sigmoid")
#             ])
#             model_instance.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#         elif model == "cnn":
#             model_instance = Sequential([
#                 Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(60, len(predictors))),
#                 MaxPooling1D(pool_size=2),
#                 Conv1D(filters=64, kernel_size=3, activation="relu"),
#                 MaxPooling1D(pool_size=2),
#                 Flatten(),
#                 Dense(50, activation="relu"),
#                 Dropout(0.3),
#                 Dense(1, activation="sigmoid")
#             ])
#             model_instance.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#         else:
#             raise ValueError("Invalid model type specified.")

#         predictions = backtest(data, model_instance, predictors, model_type=model)

#         precision = precision_score(predictions["Target"], predictions["Predictions"])
#         prediction_distribution = predictions["Predictions"].value_counts().to_dict()
#         target_distribution = (predictions["Target"].value_counts() / predictions.shape[0]).to_dict()

#         img_actual = plot_actual(predictions, graph_style=graph_style)
#         img_predicted = plot_predicted(predictions, graph_style=graph_style)
#         img_target_pie = plot_pie_chart(target_distribution, title="Actual Target Distribution")
#         img_prediction_pie = plot_pie_chart(prediction_distribution, title="Prediction Distribution")

#         return {
#             "ticker": ticker.upper(),
#             "precision": precision,
#             "prediction_distribution": prediction_distribution,
#             "target_distribution": target_distribution,
#             "image_actual": img_actual,
#             "image_predicted": img_predicted,
#             "image_target_pie": img_target_pie,
#             "image_prediction_pie": img_prediction_pie,
#             "status": "success"
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ta.momentum import RSIIndicator

def load_data(ticker, interval="1mo", start_date=None, end_date=None):
    if start_date or end_date:
        print(f"Loading data for {ticker} from {start_date} to {end_date} with interval {interval}")
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
    else:
        print(f"Loading 10 years data for {ticker} with interval {interval}")
        data = yf.Ticker(ticker).history(period="10y", interval=interval)
    
    print(f"Fetched {len(data)} rows for {ticker}")
    if data.empty:
        raise ValueError("No data found for this ticker.")
    data.index = pd.to_datetime(data.index)
    return data

def engineer_features(data, model_type="randomforest"):
    data = data.copy()
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    
    # Add RSI
    rsi = RSIIndicator(data["Close"], window=14)
    data["RSI"] = rsi.rsi()

    if model_type == "randomforest":
        available_rows = data.shape[0]
        horizons = [h for h in [2, 5, 60, 250, 1000] if h < available_rows // 2]
        new_predictors = ["RSI"]

        for horizon in horizons:
            rolling_averages = data.rolling(horizon).mean()
            ratio_col = f"Close_Ratio_{horizon}"
            trend_col = f"Trend_{horizon}"
            data[ratio_col] = data["Close"] / rolling_averages["Close"]
            data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]
            new_predictors += [ratio_col, trend_col]
        
        data.dropna(subset=[col for col in data.columns if col != "Tomorrow"], inplace=True)
        return data, new_predictors
    else:  # LSTM or CNN
        predictors = ["Close", "Volume", "Open", "High", "Low", "RSI"]
        scaler = MinMaxScaler()
        data[predictors] = scaler.fit_transform(data[predictors])
        data.dropna(subset=[col for col in data.columns if col != "Tomorrow"], inplace=True)
        return data, predictors

def process_image(image_path):
    if not image_path:
        return None
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def create_sequences(data, predictors, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[predictors].iloc[i:(i + seq_length)].values)
        y.append(data["Target"].iloc[i + seq_length])
    return np.array(X), np.array(y)

def predict_randomforest(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    return pd.DataFrame({
        "Target": test["Target"],
        "Predictions": preds
    }, index=test.index)

def predict_neural(train, test, predictors, model, seq_length=60):
    X_train, y_train = create_sequences(train, predictors, seq_length)
    X_test, y_test = create_sequences(test, predictors, seq_length)
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Not enough data to create sequences.")
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    preds = model.predict(X_test, verbose=0)
    preds = (preds >= 0.5).astype(int).flatten()
    return pd.DataFrame({
        "Target": y_test,
        "Predictions": preds
    }, index=test.index[-len(y_test):])

def predict_cnn_hybrid(train, test, predictors, model, seq_length=60, image_path=None):
    X_train, y_train = create_sequences(train, predictors, seq_length)
    X_test, y_test = create_sequences(test, predictors, seq_length)
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Not enough data to create sequences.")
    
    img_features = process_image(image_path)
    if img_features is not None:
        # Repeat image features to match sequence length
        img_features = np.repeat(img_features, X_test.shape[0], axis=0)
        model.fit([X_train, np.repeat(img_features[:1], X_train.shape[0], axis=0)], y_train, epochs=10, batch_size=32, verbose=0)
        preds = model.predict([X_test, img_features], verbose=0)
    else:
        # Use numerical data only
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        preds = model.predict(X_test, verbose=0)
    
    preds = (preds >= 0.5).astype(int).flatten()
    return pd.DataFrame({
        "Target": y_test,
        "Predictions": preds
    }, index=test.index[-len(y_test):])

def backtest(data, model, predictors, model_type="randomforest", start=None, step=250, seq_length=60, image_path=None):
    all_predictions = []

    if not start or start >= data.shape[0]:
        start = max(20, int(data.shape[0] * 0.2))

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()

        if test.empty or train.empty or len(test) < (seq_length + 5 if model_type in ["lstm", "cnn"] else 5) or len(train) < (seq_length + 20 if model_type in ["lstm", "cnn"] else 20):
            continue

        if model_type == "randomforest":
            predictions = predict_randomforest(train, test, predictors, model)
        elif model_type == "lstm":
            predictions = predict_neural(train, test, predictors, model, seq_length)
        else:  # cnn
            predictions = predict_cnn_hybrid(train, test, predictors, model, seq_length, image_path)
        all_predictions.append(predictions)

    if not all_predictions:
        raise ValueError("No valid prediction windows found. Try using a stock with more historical data.")

    return pd.concat(all_predictions)

def plot_actual(data, graph_style="line"):
    fig, ax = plt.subplots(figsize=(18, 6))
    if graph_style == "line":
        data["Target"].plot(ax=ax, color="blue", label="Actual")
    elif graph_style == "bar":
        ax.bar(data.index, data["Target"], color="blue", label="Actual")
    elif graph_style == "step":
        ax.step(data.index, data["Target"], color="blue", label="Actual")
    elif graph_style == "scatter":
        ax.scatter(data.index, data["Target"], color="blue", label="Actual", s=10)
    else:
        data["Target"].plot(ax=ax, color="blue", label="Actual")

    ax.legend()
    ax.set_title("Actual Target")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return img

def plot_predicted(data, graph_style="line"):
    fig, ax = plt.subplots(figsize=(18, 6))
    if graph_style == "line":
        data["Predictions"].plot(ax=ax, color="orange", label="Predicted")
    elif graph_style == "bar":
        ax.bar(data.index, data["Predictions"], color="orange", label="Predicted")
    elif graph_style == "step":
        ax.step(data.index, data["Predictions"], color="orange", label="Predicted")
    elif graph_style == "scatter":
        ax.scatter(data.index, data["Predictions"], color="orange", label="Predicted", s=10)
    else:
        data["Predictions"].plot(ax=ax, color="orange", label="Predicted")

    ax.legend()
    ax.set_title("Predicted Target")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return img

def plot_pie_chart(distribution, title="Distribution"):
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = [f"{k}: {v}" for k, v in distribution.items()]
    values = list(distribution.values())

    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return img

def run_model(ticker, interval="1d", start_date=None, end_date=None, graph_style="line", model="randomforest", image_path=None):
    try:
        data = load_data(ticker, interval=interval, start_date=start_date, end_date=end_date)
        data, predictors = engineer_features(data, model_type=model)

        print(f"Data after feature engineering: {len(data)} rows")

        if model == "randomforest":
            model_instance = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
        elif model == "lstm":
            model_instance = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, len(predictors))),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1, activation="sigmoid")
            ])
            model_instance.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        elif model == "cnn":
            if image_path:
                # Hybrid CNN model with image and numerical inputs
                numerical_input = Input(shape=(60, len(predictors)))
                image_input = Input(shape=(224, 224, 3))
                
                # Image branch
                base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
                base_model.trainable = False
                x1 = base_model(image_input)
                x1 = Flatten()(x1)
                x1 = Dense(128, activation="relu")(x1)
                
                # Numerical branch
                x2 = Conv1D(filters=32, kernel_size=3, activation="relu")(numerical_input)
                x2 = MaxPooling1D(pool_size=2)(x2)
                x2 = Conv1D(filters=64, kernel_size=3, activation="relu")(x2)
                x2 = MaxPooling1D(pool_size=2)(x2)
                x2 = Flatten()(x2)
                x2 = Dense(50, activation="relu")(x2)
                
                # Combine branches
                combined = Concatenate()([x1, x2])
                combined = Dense(64, activation="relu")(combined)
                combined = Dropout(0.3)(combined)
                output = Dense(1, activation="sigmoid")(combined)
                
                model_instance = Model(inputs=[numerical_input, image_input], outputs=output)
                model_instance.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            else:
                # Numerical-only CNN model
                model_instance = Sequential([
                    Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(60, len(predictors))),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=64, kernel_size=3, activation="relu"),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(50, activation="relu"),
                    Dropout(0.3),
                    Dense(1, activation="sigmoid")
                ])
                model_instance.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            raise ValueError("Invalid model type specified.")

        predictions = backtest(data, model_instance, predictors, model_type=model, image_path=image_path)

        precision = precision_score(predictions["Target"], predictions["Predictions"])
        prediction_distribution = predictions["Predictions"].value_counts().to_dict()
        target_distribution = (predictions["Target"].value_counts() / predictions.shape[0]).to_dict()

        img_actual = plot_actual(predictions, graph_style=graph_style)
        img_predicted = plot_predicted(predictions, graph_style=graph_style)
        img_target_pie = plot_pie_chart(target_distribution, title="Actual Target Distribution")
        img_prediction_pie = plot_pie_chart(prediction_distribution, title="Prediction Distribution")

        return {
            "ticker": ticker.upper(),
            "precision": precision,
            "prediction_distribution": prediction_distribution,
            "target_distribution": target_distribution,
            "image_actual": img_actual,
            "image_predicted": img_predicted,
            "image_target_pie": img_target_pie,
            "image_prediction_pie": img_prediction_pie,
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}