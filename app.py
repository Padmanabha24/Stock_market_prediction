# from flask import Flask, request, render_template
# from model import run_model

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         ticker = request.form["ticker"]
#         interval = request.form["interval"]
#         start_date = request.form.get("start_date")
#         end_date = request.form.get("end_date")
#         graph_style = request.form.get("graph_style", "line")  # default to line

#         result = run_model(ticker, interval=interval, start_date=start_date, end_date=end_date, graph_style=graph_style)
#         return render_template("index.html", result=result)
#     return render_template("index.html")



# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
from model import run_model
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/randomforest", methods=["GET", "POST"])
def randomforest():
    if request.method == "POST":
        ticker = request.form["ticker"]
        interval = request.form["interval"]
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        graph_style = request.form.get("graph_style", "line")
        result = run_model(ticker, interval=interval, start_date=start_date, end_date=end_date, graph_style=graph_style, model="randomforest")
        return render_template("RandomForest.html", result=result)
    return render_template("RandomForest.html")

@app.route("/lstm", methods=["GET", "POST"])
def lstm():
    if request.method == "POST":
        ticker = request.form["ticker"]
        interval = request.form["interval"]
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        graph_style = request.form.get("graph_style", "line")
        result = run_model(ticker, interval=interval, start_date=start_date, end_date=end_date, graph_style=graph_style, model="lstm")
        return render_template("LSTM.html", result=result)
    return render_template("LSTM.html")

@app.route("/cnn", methods=["GET", "POST"])
def cnn():
    if request.method == "POST":
        ticker = request.form["ticker"]
        interval = request.form["interval"]
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        graph_style = request.form.get("graph_style", "line")
        chart_image = request.files.get("chart_image")
        image_path = None
        if chart_image and chart_image.filename:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], chart_image.filename)
            chart_image.save(image_path)
        result = run_model(ticker, interval=interval, start_date=start_date, end_date=end_date, graph_style=graph_style, model="cnn", image_path=image_path)
        return render_template("CNN.html", result=result)
    return render_template("CNN.html")

if __name__ == "__main__":
    app.run(debug=True)