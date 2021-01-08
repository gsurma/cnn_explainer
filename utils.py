import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

def preprocess_image(image):
    image = transforms.functional.to_tensor(image)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
