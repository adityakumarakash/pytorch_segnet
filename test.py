import argparse
import google.protobuf.text_format as txtf
import numpy as np
import os
import torch

from collections import OrderedDict
from datasets.camvid_dataset import CamvidDataset
from models.segnet import Segnet
from skimage import io
from torchvision import transforms
from utils import config_pb2


def clean_state_dict(state_dict):
    # Cleans the state dict by removing the module prefix from the keys.
    new_state_dict = OrderedDict()
    prefix = "module."
    for key, value in state_dict.items():
        new_key = key[len(prefix):]  # removes prefix  
        new_state_dict[new_key] = value
    return new_state_dict

def test(config):
    model_path = config.test.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy loader for this datatset.
    dummy_loader = CamvidDataset()
    model = Segnet(dummy_loader.num_classes).to(device)

    model_dict = torch.load(model_path)
    print("Loaded model from :", model_path)
    model.load_state_dict(clean_state_dict(model_dict["model_state"]))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])    
    
    if config.test.evaluate_on_test_split:
        # TODO
        print("Not yet implemented")
    else:
        image_path = config.test.image_path
        output_path = config.test.output_path

        image = io.imread(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        
        output = model(image)
        pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
        dataset = CamvidDataset()
        segmentation = dataset.color_segmentation(pred)
        io.imsave(output_path, segmentation)


if __name__ == "__main__":
    # Parses the command line arguments
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file for model",
    )
    args = parser.parse_args()

    # Creates config using the arguments.
    config = config_pb2.Config()
    with open(args.config) as f:
        txtf.Merge(f.read(), config)

    test(config)
    
