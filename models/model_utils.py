import torch
import argparse
import sys
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as functional


def get_output_tensor(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    x = preprocess(img)
    batch = x.unsqueeze(0)
    return batch


def get_input_tensor(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    x = functional.to_tensor(image)
    return x.view(1, 3, 224, 224)


def run_model(model, model_file, input_tensor, eval=True):
    model.eval()

    if eval:
        with torch.no_grad():
            out = model(input_tensor)
        print(input_tensor)
        print(out[0].softmax(dim=0).topk(5, dim=0))
    else:
        traced_script_model = torch.jit.trace(model, input_tensor)
        traced_script_model.save(model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str,
                        help='Mode for the model options [eval, create].', default="eval",
                        required=True)
    parser.add_argument('-model', type=str,
                        help='Model to create or evaluate.',
                        default="resnet_50", required=False)
    parser.add_argument('-input', type=str,
                        help='Name of input to apply to model.',
                        default="grace_hopper_517x606.jpg",
                        required=False)

    args = parser.parse_args(sys.argv[1:])

    img = args.input
    if args.mode == "eval":
        x = get_output_tensor(img)
        eval = True
    else:
        x = get_input_tensor(img)
        eval = False

    model_options = {
        "resnet_18": models.resnet18(pretrained=True),
        "resnet_34": models.resnet34(pretrained=True),
        "resnet_50": models.resnet50(pretrained=True)
    }
    name = args.model + "_pretrained.pt"
    model = model_options[args.model]
    run_model(model, name, x, eval)
