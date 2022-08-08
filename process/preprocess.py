import cv2
import numpy as np
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from tqdm import tqdm
import copy
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    empty_cuda_cache,
)
from craft_text_detector.file_utils import rectify_poly
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from minimumboundingbox.MinimumBoundingBox import MinimumBoundingBox
import os
import numpy as np

config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
# config['weights'] = '../synthesis_data/ocr_data/test.pth'
detector = Predictor(config)
craft_net = load_craftnet_model(cuda = True)
refine_net = load_refinenet_model(cuda= True)

def order_points(pts):
    # sort points clockwise
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # function transform image with 4 points
    rect = order_points(pts)
    tl, tr, br, bl = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return M, warped

def rotate_points(box, M):     # Rotate each point in 4 points: [x1, y1, x2, y2... x4, y4]
    # function rotate point with given matrix in cv2.getPerspectiveTransform built in function
    box_np = np.array(box).astype(np.float)
    box_np = np.rint(box_np).astype(np.int32)   # np.rint == round int
    # print(box_np.shape)
    box_np = box_np.reshape(-1, 2)  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], return shape (4, 2)
        # add ones
    ones = np.ones(shape=(len(box_np), 1))  # [1, 1, 1, 1]
    points_ones = np.hstack([box_np, ones]) # [[x1, y1, 1],
                                                #  [x2, y2, 1],
                                                #  [x3, y3, 1],
                                                #  [x4, y4, 1]], return shape (4, 3)
        # transform points
    transformed_points = M.dot(points_ones.T).T
    transformed_points2 = transformed_points.reshape(-1)
    transformed_points2 = np.rint(transformed_points2)
    transformed_points2 = transformed_points2.astype(int)
    transformed_points2 = transformed_points2.reshape(-1,3)
    transformed_points2 = np.delete(transformed_points2, np.s_[-1], axis=1)
    return transformed_points2

def draw_bbox_poly(box):
    pass

############################
def extract_text_box(img):
    img = copy.deepcopy(img)
    prediction_result = get_prediction(
        image=img,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.8,
        link_threshold=0.4,
        low_text=0.3,
        cuda=True,
        long_size=1280
        )
    result_list = []
    for region in prediction_result["boxes"]:
        x1, y1= np.min(region, axis = 0)
        x2, y2= np.max(region, axis = 0)
        result_list.append(([int(x1), int(y1), int(x2), int(y2)],np.array(rectify_poly(img, region))))
    return result_list

def extract_text_box_v2(img):
    img = copy.deepcopy(img)
    prediction_result = get_prediction(
        image=img,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.8,
        link_threshold=0.4,
        low_text=0.3,
        cuda=True,
        long_size=1280
        )
    result_list = []
    for region in prediction_result["boxes"]:
        result_list.append((region,np.array(rectify_poly(img, region))))
    return result_list

def convex_hull_processing(img):
    result = extract_text_box_v2(img)
    coords = result[0][0]
    for i in range(1,len(result)):
        coords = np.append(coords, result[i][0], axis= 0)
    hull = ConvexHull(coords)
    points = tuple()
    for idx in hull.vertices:
        points += ((coords[idx, 0], coords[idx, 1]),)
    points = tuple(points)
    # print(points)
    bounding_box = MinimumBoundingBox(points)  # returns namedtuple
    rect = list(bounding_box.corner_points)
    # print(rect)
    rect = list(map(lambda x: list(x), rect))
    rect = np.array(rect)
    M, warped = four_point_transform(img, rect)
    # cv2.imwrite(os.path.join('/mnt/disk1/vaipe-hiepnm/receipt-ai/static/img', 'result.png'), warped)
    transformed_points = []
    for i in range(len(result)):
        tmp = rotate_points(result[i][0], M)
        # print(tmp, result[i][0])
        x1, y1 = np.min(tmp, axis=0)
        x2, y2 = np.max(tmp, axis=0)
        transformed_points.append(([int(x1), int(y1), int(x2), int(y2)], np.array(rectify_poly(warped, tmp))))
    # print(transformed_points)
    return warped, transformed_points

def recursive_cnn_processing(img):
    pass
def ocr_model_predict(img, ocr_model):
    return ocr_model.predict(img)

def pre_processing(img_bytes, filenames):
    processed_annos = []
    for img_p, fname in zip(img_bytes, filenames):
        if isinstance(img_p, str):
            img = cv2.imread(img_p)
        else:
            imgarr = np.asarray(bytearray(img_p), dtype=np.uint8)
            img = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotation = []
        print("Begin extracting box -------")
        recified_boxes = extract_text_box(img)
        print("Extract box done ----------")
        print(recified_boxes[0][0])
        if len(recified_boxes) > 0:
            recified_boxes = sorted(recified_boxes, key= lambda x: (x[0][1], x[0][0]))
        for count, bbox in enumerate(tqdm(recified_boxes,desc="Processing...")):
            img_box = bbox[1]
            img_box = Image.fromarray(np.uint8(img_box)).convert('RGB')
            pred = ocr_model_predict(img_box,detector)
            annotation.append(dict(
                id= count,
                text = pred,
                label= 'other',
                box=list(map(int, bbox[0])),
                linking=[]
            ))
        processed_annos.append({"fname": fname, "anno": annotation})
    return processed_annos

def pre_processing_v2(img_bytes, filenames):
    processed_annos = []
    for img_p, fname in zip(img_bytes, filenames):
        if isinstance(img_p, str):
            img = cv2.imread(img_p)
        else:
            imgarr = np.asarray(bytearray(img_p), dtype=np.uint8)
            img = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotation = []
        print("Begin extracting box -------")
        warped_img, recified_boxes = convex_hull_processing(img)
        print("Extract box done ----------")
        # print(recified_boxes[0])
        if len(recified_boxes) > 0:
            recified_boxes = sorted(recified_boxes, key= lambda x: (x[0][1], x[0][0]))
        for count, bbox in enumerate(tqdm(recified_boxes,desc="Processing...")):
            img_box = bbox[1]
            img_box = Image.fromarray(np.uint8(img_box)).convert('RGB')
            pred = ocr_model_predict(img_box,detector)
            annotation.append(dict(
                id= count,
                text = pred,
                label= 'other',
                box=list(map(int, bbox[0])),
                linking=[]
            ))
        cv2.imwrite(os.path.join('/mnt/disk1/doan/vaipe-hiepnm/receipt-ai/static/img/',"warped_"+ fname), warped_img)
        processed_annos.append({"fname": fname, "anno": annotation})
    return processed_annos

def pre_processing_v3(img_bytes, filenames):
    pass