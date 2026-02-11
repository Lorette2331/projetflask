from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def presentation():
    return render_template("index.html")

@app.route("/etudes")
def etudes():
    return render_template("etudes.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/map")
def map():
    return render_template("map.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
