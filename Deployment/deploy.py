import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'


n_classes=5
class_index = {
    0:'Light Diseases and Disorders of Pigmentation',
    1:'Acne and Rosacea Photos',
    2:'Poison Ivy Photos and other Contact Dermatitis',
    3:'Atopic Dermatitis Photos',
    4:'Hair Loss Photos Alopecia and other Hair Diseases',
}

PATH='best_model.pth'
import torchvision.models as models
import torch.nn as nn


model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, n_classes)

model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
       [ transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.float() / 255),]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)

    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return predicted_idx, class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()