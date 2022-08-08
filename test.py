import argparse
import glob
import json
import os.path as osp
import time
import cv2
import torch

from torch_geometric.data import DataLoader
from tqdm import tqdm
import copy
from dataset.dataset import Receipt
from model import BERTxGCN, BERTxSAGE, BERTxGAT
from process import post_processing4, post_processing2
from utils.metrics import MetricTracker
from utils import LABELS, expand


parser = argparse.ArgumentParser(description="PyTorch BERT-GCN")
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="disables CUDA training")
parser.add_argument("--batch-size", type=int, default=1, metavar="N",
                    help="input batch size for training (default: 2)")
parser.add_argument("--test-folder", type=str,
                    default="dataset/test_data/",
                    help="test folder path")
parser.add_argument("--weight", type=str,
                    default="./logs/saved/model_best.pth",
                    help="model weight path")
parser.add_argument("--seed", type=int, default=42, metavar="S",
                    help="random seed (default: 42)")

if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(args.seed)

    test_files = glob.glob(args.test_folder + "/*.json")

    test_dataset = Receipt(test_files)
    print(f"Number of test set: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = BERTxSAGE()
    # model = BERTxGAT()
    # print(model)
    model = torch.load(args.weight)
    if args.cuda:
        # Move model to GPU.
        model.cuda()
    else:
        model.cpu()
    start_time = time.time()
    model.eval()
    font = cv2.FONT_HERSHEY_SIMPLEX
    metric = MetricTracker(labels=LABELS)
    for data in tqdm(test_loader, desc="Test"):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            output = model(data)            # sum up batch loss
            # get the index of the max log-probability
            preds = output.data.max(1, keepdim=True)[1]
            preds.squeeze_()
            for i in range(len(data.imname)):
                bboxes = []
                pred_ = preds[data.batch == i].cpu().numpy()
                bbox_ = data.bbox[data.batch == i]
                imname = data.imname[i].replace("json", "jpg")
                im_path = osp.join(
                    args.test_folder, imname)
                if not osp.isfile(im_path):
                    print("No file exists", im_path)
                    continue
                img = cv2.imread(im_path)
                img = expand(img)
                h, w, _ = img.shape
                h, w = 2000, 2000
                text_ = data.text[i]
                for bbox, pred, text in zip(bbox_, pred_, text_):
                    pred = pred.item()
                    x0, x1 = int(bbox[0]*w), int(bbox[2]*w)
                    y0, y1 = int(bbox[1]*h), int(bbox[3]*h)
                    bboxes.append(dict(
                        bbox=[x0, y0, x1, y1], pred=LABELS[int(
                            pred)], text=text
                    ))
                bboxes_copy = copy.deepcopy(bboxes)
                result = post_processing4(bboxes_copy)
                # for j, bbox in enumerate(bboxes):
                #     pred_[j] = LABELS.index(bbox["pred"])
                # print(bboxes)
                y_gt_ = data.y.data[data.batch == i].cpu()
                pred_ = torch.from_numpy(pred_)
                metric.update(pred_, y_gt_.view_as(pred_))

                for j, item in enumerate(bboxes):
                    bbox = item["bbox"]
                    if LABELS[int(pred_[j])] != 'other':
                        img = cv2.rectangle(
                            img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (36, 255, 12), 1)
                        img = cv2.putText(img, LABELS[int(pred_[j])], (bbox[2] + 15, bbox[1]+15), font,
                                          0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # img = cv2.putText(img, item["pred"], (bbox[0], bbox[3]), font,
                    #         0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        # img = cv2.putText(img, LABELS[int(y_gt_[j])], (bbox[2] +15, bbox[1]+25), font,
                        #                 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                with open(osp.join("predicted", imname[:-5] + ".json"), "w", encoding='utf8') as f:
                    json.dump(result, f, indent=4,  ensure_ascii=False)
                cv2.imwrite(osp.join("predicted", imname[:-5] + ".jpg"), img)
    print(f"Classification Report:")
    print(metric.compute())
    print(time.time() - start_time)
