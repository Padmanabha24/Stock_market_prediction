import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import io
import base64
import pytz  
  
def load_data(ticker, interval="1mo", start_date=None, end_date=None):
    # Use yfinance start/end params if dates are provided
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

def engineer_features(data):
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    data = data.copy()

    available_rows = data.shape[0]
    horizons = [h for h in [2, 5, 60, 250, 1000] if h < available_rows // 2]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()
        ratio_col = f"Close_Ratio_{horizon}"
        trend_col = f"Trend_{horizon}"

        data[ratio_col] = data["Close"] / rolling_averages["Close"]
        data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_col, trend_col]

    data.dropna(subset=[col for col in data.columns if col != "Tomorrow"], inplace=True)
    return data, new_predictors

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    return pd.DataFrame({
        "Target": test["Target"],
        "Predictions": preds
    }, index=test.index)

def backtest(data, model, predictors, start=None, step=250):
    all_predictions = []

    if not start or start >= data.shape[0]:
        start = max(20, int(data.shape[0] * 0.2))

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()

        # Relaxed sizes for debugging
        if test.empty or train.empty or len(test) < 5 or len(train) < 20:
            continue

        predictions = predict(train, test, predictors, model)
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
        data["Target"].plot(ax=ax, color="blue", label="Actual")  # fallback

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
        data["Predictions"].plot(ax=ax, color="orange", label="Predicted")  # fallback

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

def run_model(ticker, interval="1d", start_date=None, end_date=None, graph_style="line"):
    try:
        data = load_data(ticker, interval=interval, start_date=start_date, end_date=end_date)
        data, predictors = engineer_features(data)

        print(f"Data after feature engineering: {len(data)} rows")

        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
        predictions = backtest(data, model, predictors)

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
