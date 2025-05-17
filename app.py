from flask import Flask, request, render_template
from model import run_model

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form["ticker"]
        interval = request.form["interval"]
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        graph_style = request.form.get("graph_style", "line")  # default to line

        result = run_model(ticker, interval=interval, start_date=start_date, end_date=end_date, graph_style=graph_style)
        return render_template("index.html", result=result)
    return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)
