import torch
from torchvision import transforms
from torch import nn
def img_feature_extractor(model, img, embedding_dim = 256):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = img.convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    model.fc = nn.Linear(model.fc.in_features, embedding_dim)
    with torch.no_grad():
        output = model(input_batch)
    return output