from commentModel import Model
from flask import Flask, render_template, request
import torch
from transformers import GPT2Tokenizer

app = Flask(__name__, template_folder = "./static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def process_comment():
    comment = request.form['comment']
    prediction = predict(comment)
    isToxic = (prediction < 0.5)
    print(isToxic)
    return render_template("index.html", isToxic=isToxic, prediction="{:.2f}".format(prediction))

def predict(comment):
    model = Model()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoded = tokenizer(comment, return_tensors='pt')
    x = encoded["input_ids"].float()
    x = torch.nn.functional.pad(x,(1,468-x.size(dim=1)))
    return model(x).item()

app.run(port=7000)