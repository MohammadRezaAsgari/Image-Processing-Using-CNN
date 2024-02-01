from django.shortcuts import render
from rest_framework.views import APIView
from django.http import JsonResponse
from django.core.files.storage import default_storage
import torch
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models
from torchvision import transforms

class ProcessApiView(APIView):

    def post(self, request):

        image_file = request.FILES.get('image')

        if image_file:
            model = FaceRecognitionModel(num_classes=15)
            directory_path = os.getcwd()
            model.load_state_dict(torch.load(directory_path + "/core/fine_face_recognition_model.pth"))
            model.eval()

            transform = transforms.Compose([
                # transforms.RandomRotation(20),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
            ])

            def predict_class(image):
                transformed_image = transform(image)
                transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    output = model(transformed_image)
                    _, predicted_class = torch.max(output, 1)
                return predicted_class.item()
            
            def preprocess_image(image_path, target_size=(256, 256)):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                img_rgb = cv2.resize(img_rgb, target_size)
                img_rgb = img_rgb / 255.0
                img_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))
                return img_pil

            
            destination_path = default_storage.path(image_file.name)
            with default_storage.open(destination_path, 'wb') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            
            predicted_class = predict_class(preprocess_image(destination_path))

            index_to_class = {v: k for k, v in class_to_label.items()}
            predicted_label = index_to_class[predicted_class]

            print(f"Predicted Class: {predicted_label}")

            return JsonResponse({'class': predicted_label})
        else:
            return JsonResponse({'message': 'No file was uploaded'}, status=400)
        

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()

        # Load pre-trained ResNet-18 model
        resnet18 = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.resnet_layers = nn.Sequential(*list(resnet18.children())[:-1])

        # Add your custom layers with increased capacity
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through ResNet layers
        x = self.resnet_layers(x)
        x = F.leaky_relu(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.fc3(x)
        return x

     

class_to_label = {
    "chavoshi": 0,
    "shajarian": 1,
    "khaliq": 2,
    "radan": 3,
    "bayati": 4,
    "kianafshar": 5,
    "alidoosti": 6,
    "qaforian": 7,
    "razavian": 8,
    "daei": 9,
    "attaran": 10,
    "beiranvand": 11,
    "dolatshahi": 12,
    "esfahani": 13,
    "hoceini": 14,
} 
