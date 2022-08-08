import os
import torch
import json
import numpy as np
import cv2
import base64
import copy
from torch_geometric.data import DataLoader
from flask import Flask, render_template, request
from flask_fontawesome import FontAwesome
from werkzeug.utils import secure_filename
from dataset.dataset import Receipt
from model import BERTxSAGE
from process import pre_processing_v2, pre_processing, post_processing4
from utils import LABELS
from flask import jsonify, json
import jellyfish
app = Flask(__name__)
fa = FontAwesome(app)
app.config['SECRET_KEY'] = '192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/img')
BATCH_SIZE = 1
font = cv2.FONT_HERSHEY_SIMPLEX
model = BERTxSAGE()
model = torch.load("./logs/saved/model_best.pth")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_files():
    return render_template('home.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        files = request.files.getlist('file')
        option = request.form.getlist('preprocessing')
        result = []
        print(files)
        if files:
            img_bytes = [file.read() for file in files]
            filenames = [secure_filename(file.filename) for file in files]
            if option[0] == 'none':
                annotations = pre_processing(img_bytes, filenames)
            elif option[0] == 'convex_hull':
                annotations = pre_processing_v2(img_bytes, filenames)
            data = Receipt(annotations)
            data_loader = DataLoader(data, batch_size=BATCH_SIZE)
            for batch in data_loader:
                batch.to(device)
                with torch.no_grad():
                    output = model(batch)            # sum up batch loss
                    # get the index of the max log-probability
                    preds = output.data.max(1, keepdim=True)[1]
                    preds.squeeze_()
                    preds = preds.cpu().numpy()
                    h, w = 1000, 1000
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
                        if option[0] == 'none':
                            imgarr = np.asarray(bytearray(img_bytes[i]), dtype=np.uint8)
                            img = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        elif option[0] == 'convex_hull':
                            img = cv2.imread(os.path.join(UPLOAD_FOLDER,'warped_'+files[i].filename))
                        ## update 15/4 change label of wrong diagnose to 'other' 
                        flag_first_drugname = False
                        for item in bboxes:
                            if item['pred'] == 'diagnose' and flag_first_drugname:
                                item['pred'] = 'other'
                            if jellyfish.jaro_distance(item['text'], 'Chuẩn đoán khác') > 0.8:
                                item['pred'] = 'other'
                            if item['pred'] == 'drugname':
                                flag_first_drugname = True
                        ################
                        for item in bboxes:
                            print(item)
                            bbox = item["bbox"]
                            if item['pred'] != 'other':
                                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (36, 255, 12), 1)
                                img = cv2.putText(img, item['pred'], (bbox[2] + 15, bbox[1]+15), font,
                                          0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        height, width , _ = img.shape
                        cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'result_'+files[i].filename), img)
                        prediction = post_processing4(bboxes)
                        print(prediction)
                        result.append(dict(
                            name = 'img/result_'+batch.imname[i], 
                            result = prediction,
                            height = height,
                            width = width
                        ))
            return render_template('home.html', response = result)
    # return jsonify(result) 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8123, debug=False)