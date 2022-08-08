import os
from flask import Flask, render_template, request, flash
from flask_fontawesome import FontAwesome
from werkzeug.utils import secure_filename
from dataset.dataset import Receipt
from model import BERTxGCN, BERTxSAGE
from process import pre_processing, post_processing2
from utils import LABELS, expand
import torch
from torch_geometric.data import DataLoader
import json
import numpy as np
import cv2
import base64
import copy
app = Flask(__name__)
fa = FontAwesome(app)
app.config['SECRET_KEY'] = 'demo ocr'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
BATCH_SIZE = 1
font = cv2.FONT_HERSHEY_SIMPLEX
model = BERTxSAGE()
model = torch.load("./logs/saved/model_best.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict/')
def upload_files():
    return render_template('index.html')
@app.route('/predict/',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        files = request.files.getlist('file')
        # print(files)
        if files:
            img_bytes = [file.read() for file in files]
            filenames = [secure_filename(file.filename) for file in files]
            annotations = pre_processing(img_bytes, filenames)
            data = Receipt(annotations)
            data_loader = DataLoader(data, batch_size=BATCH_SIZE)
            result = {}
            for batch in data_loader:
                batch.to(device)
                with torch.no_grad():
                    output = model(batch)            # sum up batch loss
                    # get the index of the max log-probability
                    preds = output.data.max(1, keepdim=True)[1]
                    preds.squeeze_()
                    preds = preds.cpu().numpy()
                    h, w = 2000, 2000
                    batch.to(torch.device("cpu"))
                    for i in range(len(batch.imname)):
                        bboxes = []
                        bbox_ = batch.bbox[batch.batch==i]
                        pred_ = preds[batch.batch==i]
                        text_ = batch.text[i]
                        for bbox, pred, text in zip(bbox_, pred_, text_):
                            pred = pred.item()
                            x0, x1 = int(bbox[0]*w), int(bbox[2]*w)
                            y0, y1 = int(bbox[1]*h), int(bbox[3]*h)
                            bboxes.append(dict(
                                bbox=[x0, y0, x1, y1], pred=LABELS[pred], text=text
                            ))
                        prediction = post_processing2(bboxes)
                        result[batch.imname[i]] = prediction

            filepaths = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in filenames]
            for file, path in zip(files, filepaths):
                file.save(path)
            response = app.response_class(
                response=json.dumps(result, indent=4,  ensure_ascii=False),
                status=200,
                mimetype='application/json'
            )
            return response

        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return app.response_class(
                response=json.dumps('Only allow types txt, pdf, png, jpg, jpeg, gif'),
                status=500,
                mimetype='application/json'
            )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8123, debug=False)