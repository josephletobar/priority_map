from ultralytics.engine.results import Results
import numpy as np
import cv2
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_image(segmented_object):

    image = Image.fromarray(segmented_object)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features = image_features.cpu().numpy().squeeze().tolist()
    return image_features

def embed_text(text):

    text_inputs = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features = text_features.cpu().numpy()[0]
    return text_features