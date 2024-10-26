from commentModel import Model
from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer

app = Flask(__name__, template_folder = "./static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def process_comment():
    comment = request.form['comment']
    prediction = predict(comment)
    isToxic = (prediction > 0.5)
    print(isToxic)
    return render_template("index.html", isToxic=isToxic, prediction="{:.2f}".format(prediction))

def predict(comment):
    tokenizer = AutoTokenizer.from_pretrained('comment_text', pad_token="~", unk_token="~")
    model = Model(pad_id=tokenizer.pad_token_id)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    encoded = tokenizer.encode_plus(comment, return_tensors='pt', max_length=100, truncation=True, padding='max_length')
    x = encoded["input_ids"].long()
    return model(x).item()

app.run(port=7000)