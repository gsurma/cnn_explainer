import torch
import numpy as np
import cv2

class Activations():

    def __init__(self, target_layer, target_layer_names):
        self.target_layer = target_layer
        self.target_layer_names = target_layer_names

    def __call__(self, in_out):
        outputs = []
        for name, module in self.target_layer._modules.items():
            in_out = module(in_out)
            if name in self.target_layer_names:
                outputs += [in_out]
        return outputs, in_out
    
class Gradients():

    def __init__(self, target_layer, target_layer_names):
        self.target_layer = target_layer
        self.target_layer_names = target_layer_names

    def save_gradients(self, grad):
        self.gradients.append(grad)
        
    def get_gradients(self):
        return self.gradients

    def __call__(self, in_out):
        self.gradients = []
        for name, module in self.target_layer._modules.items():
            if name in self.target_layer_names:
                in_out.register_hook(self.save_gradients)
    

class LayerExtractor():

    def __init__(self, model, target_layer, target_layer_names):
        self.model = model
        self.target_layer = target_layer
        
        self.activations = Activations(self.target_layer, target_layer_names)
        self.gradients = Gradients(self.target_layer, target_layer_names)

    def get_gradients(self):
        return self.gradients.get_gradients()

    def __call__(self, in_out):
        for name, module in self.model._modules.items():
            if module == self.target_layer:
                activations, in_out = self.activations(in_out)
                self.gradients(in_out)
            elif "avgpool" in name.lower():
                in_out = module(in_out)
                in_out = in_out.view(in_out.size(0),-1)
            else:
                in_out = module(in_out)
        return activations, in_out

class GradCam:
    
    def __init__(self, model, target_layer, target_layer_names):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.layer_extractor = LayerExtractor(self.model, self.target_layer, target_layer_names)

    def forward(self, input_image):
        return self.model(input_image)

    def __call__(self, input_image, target_class=None):

        activations, output = self.layer_extractor(input_image)

        if target_class == None: #if not explicitly set, setting the top prediction
            target_class = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_class] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output)
 
        self.target_layer.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = self.layer_extractor.get_gradients()[-1].cpu().data.numpy()
        
        activations = activations[-1]
        activations = target.cpu().data.numpy()[0, :]

        gradients = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, gradient in enumerate(gradients):
            cam += gradient * activations[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
