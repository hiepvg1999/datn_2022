import time
import aiofiles
from typing import List
import torch
from fastapi import FastAPI, File, UploadFile
from torch_geometric.data import DataLoader

from dataset.dataset import Receipt
from model import BERTxGCN, BERTxSAGE
from process import post_processing3, pre_processing
from utils import LABELS, expand

BATCH_SIZE = 4
app = FastAPI()

model = BERTxSAGE()
model = torch.load("./logs/saved/model_best.pth")
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)
@app.get("/ping")
def pong():
    return {"ping": "pong!"}

@app.post("/upload")
async def parse(img: UploadFile = File(...)):
    async with aiofiles.open("tmp/" + img.filename, 'wb+') as out:
        await out.write(img.file.read())  # async write
    return {img.filename}

@app.put("/predict")
async def parse(imgs: List[UploadFile] = File(...)):
    """
    receipt: image file
    """
    img_bytes = [img.file.read() for img in imgs]
    filenames = [img.filename for img in imgs]
    annotations = pre_processing(img_bytes, filenames)
    t0 = time.time()
    data = Receipt(annotations)
    print(time.time() - t0)
    print(data[0])
    data_loader = DataLoader(data, batch_size=BATCH_SIZE)
    response = {}
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
                prediction = post_processing3(bboxes)
                response[batch.imname[i]]=prediction
    return response