# Main script for running the denoising pipeline

import torch
import numpy as np
from PIL import Image
import skimage

from .model import DenoisingNetwork
from .utils import add_noise, pixel_unshuffle, pixel_shuffle
from .train import train_model, train_model_2
from .evaluate import evaluate_results, plot_results
from .config import DEVICE, CHAN_EMBED, MAX_EPOCHS


def input_image_logic(img):
    img = img / np.max(img)
    input_image = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()
    return input_image


def output_image_logic(img):
    restored_img = img.squeeze(0).squeeze(0).detach().cpu().numpy()
    return restored_img


def noise2detail_run(noisy_img):
    noisy_img = input_image_logic(noisy_img)

    # Stage 1: Initial denoising
    model = DenoisingNetwork(noisy_img.shape[1], CHAN_EMBED).to(DEVICE)
    train_model(model, noisy_img)

    with torch.no_grad():
        denoised_img_1 = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    # Stage 2: Ensemble with pixel shuffling
    with torch.no_grad():
        input_2 = pixel_unshuffle(noisy_img, 2)
        pseudo1 = torch.clamp(input_2 - model(input_2), 0, 1)
        pseudo1 = pixel_shuffle(pseudo1, 2)
        pseudo1 = torch.clamp(pseudo1 - model(pseudo1), 0, 1)

        input_4 = pixel_unshuffle(noisy_img, 4)
        pseudo2 = torch.clamp(input_4 - model(input_4), 0, 1)
        pseudo2 = pixel_shuffle(pseudo2, 4)
        pseudo2 = torch.clamp(pseudo2 - model(pseudo2), 0, 1)

        denoised_img_2 = (denoised_img_1 + pseudo1 + pseudo2) / 3

    # Stage 3: Refine with second model
    model2 = DenoisingNetwork(noisy_img.shape[1], CHAN_EMBED).to(DEVICE)
    train_model_2(model2, denoised_img_2, noisy_img)

    with torch.no_grad():
        denoised_img_3 = torch.clamp(model2(denoised_img_2), 0, 1)

    return output_image_logic(denoised_img_3)