from .utils import *
from .loss import *
from .siren import INF
from .color import rgb2hsv_torch, hsv2rgb_torch


def input_image_logic(img):
    """
    Reads a grayscale image as numpy array and prepares it for CoLIE processing.
    """
    img = img / np.max(img)
    # creates three channel version of the image (required for CoLIE)
    input_image = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float().cuda()
    return input_image


def output_image_logic(img):
    """
    Receives the CoLIE output and prepares it for saving and Gradio vizualisation.
    """
    restored_img = img.squeeze(0).mean(dim=0).detach().cpu().numpy()
    return restored_img


def colie_run(img, beta, gamma, delta, L, alpha=1, epochs=100, down_size=256, window=1):
    img_rgb = input_image_logic(img)

    img_hsv = rgb2hsv_torch(img_rgb)

    img_v = get_v_component(img_hsv)
    img_v_lr = interpolate_image(img_v, down_size, down_size)
    coords = get_coords(down_size, down_size)
    patches = get_patches(img_v_lr, window)


    img_siren = INF(patch_dim=window**2, num_layers=4, hidden_dim=256, add_layer=2)
    img_siren.cuda()

    optimizer = torch.optim.Adam(img_siren.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)

    l_exp = L_exp(16,L)
    l_TV = L_TV()

    for epoch in range(epochs):
        img_siren.train()
        optimizer.zero_grad()

        illu_res_lr = img_siren(patches, coords)
        illu_res_lr = illu_res_lr.view(1,1,down_size,down_size)
        illu_lr = illu_res_lr + img_v_lr

        img_v_fixed_lr = (img_v_lr) / (illu_lr + 1e-4)

        loss_spa = torch.mean(torch.abs(torch.pow(illu_lr - img_v_lr, 2)))
        loss_tv  = l_TV(illu_lr)
        loss_exp = torch.mean(l_exp(illu_lr))
        loss_sparsity = torch.mean(img_v_fixed_lr)


        loss = loss_spa * alpha + loss_tv * beta + loss_exp * gamma + loss_sparsity * delta
        loss.backward()
        optimizer.step()


    img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v)
    img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
    img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)

    return output_image_logic(img_rgb_fixed)