import torch
from .model import LUCYD
import os 
import numpy as np

def input_image_logic(img):
    img = img / np.max(img)
    input_image = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()
    return input_image


def output_image_logic(img):
    restored_img = img.squeeze(0).squeeze(0).detach().cpu().numpy()
    return restored_img


def lucyd3d_run(noisy_img, weights):

    noisy_img = input_image_logic(noisy_img)

    WEIGHT = os.path.join(os.path.dirname(__file__), weights)

    model = LUCYD().cuda()
    model.load_state_dict(torch.load(WEIGHT))
    model.eval()

    with torch.no_grad():
        out, _, _ = model(noisy_img)

    return output_image_logic(out)