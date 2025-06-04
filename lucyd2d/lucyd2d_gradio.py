import torch
from .model import LUCYD

def lucyd2d_run(noisy_img):
    model = LUCYD().cuda()
    model.load_state_dict(torch.load('./lucyd2d.pth'))
    model.eval()

    with torch.no_grad():
        out = model(noisy_img)

    return out