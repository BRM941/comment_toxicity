from flask import Flask, render_template
import requests

app = Flask(__name__, template_folder = "./static")

@app.route("/")
def index():
    return render_template("index.html")

app.run(port=7000)