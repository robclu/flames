import torch
from torchvision import models

import PIL.Image as image
import torchvision.transforms.functional as functional


def get_input_tensor():
    image_path = 'grace_hopper_517x606.jpg'
    img = image.open(image_path)
    img = img.resize((224, 224))
    x = functional.to_tensor(img)
    return x.view(1, 3, 224, 224)


def create_resnet_50(input_tensor):
    model = models.resnet50(pretrained=True)
    model.eval()

    traced_script_model = torch.jit.trace(model, input_tensor)
    traced_script_model.save('resnet50.pt')


def eval_resnet_50(input_tensor):
    model = models.resnet50(pretrained=True)
    model.eval()
    out = model.forward(input_tensor)
    print(out)


if __name__ == '__main__':
    input = get_input_tensor()
    eval_resnet_50(input)
