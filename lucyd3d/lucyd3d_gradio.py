import torch
from .model import LUCYD

def lucyd3d_run(noisy_img):
    model = LUCYD().cuda()
    model.load_state_dict(torch.load('lucyd3d.pth'))
    model.eval()

    with torch.no_grad():
        out = model(noisy_img)

    return out