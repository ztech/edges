import torch


def interp(img: torch.Tensor, h: int, w: int, x: float, y: float):
    # return img[x,y] via bilinear interpolation
    x = min(max(x, 0), w - 1.001)
    y = min(max(y, 0), h - 1.001)
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    dx0 = x - x0
    dy0 = y - y0
    dx1 = 1 - dx0
    dy1 = 1 - dy0
    return (
        img[x0, y0] * dx1 * dy1
        + img[x1, y0] * dx0 * dy1
        + img[x0, y1] * dx1 * dy0
        + img[x1, y1] * dx0 * dy0
    )


def nms(img: torch.Tensor, ori: torch.Tensor, r=1, s=5, m=1.01):
    h, w = img.shape
    dst_img = torch.zeros_like(img)
    # suppress edges where edge is stronger in orthogonal direction
    for x in range(w):
        for y in range(h):
            if img[x, y] == 0:
                continue
            tmp_pixel = img[x, y] * m
            coso = torch.cos(ori[x, y])
            sino = torch.sin(ori[x, y])
            for d in range(-r, r + 1):
                if d == 0:
                    continue
                interp_pixel = interp(img, h, w, x + d * coso, y + d * sino)
                if tmp_pixel < interp_pixel:
                    dst_img[x, y] = 0
                    break

    # suppress noisy edge estimates near boundaries
    s = min(s, w / 2, h / 2)
    dst_img[:s, :] = (torch.tensor(range(s)) / s).repeat(h, 1).T
    dst_img[-s:, :] = (torch.tensor(range(s)) / s).repeat(h, 1).flip(1).T
    dst_img[:, :s] = (torch.tensor(range(s)) / s).repeat(w, 1)
    dst_img[:, -s:] = (torch.tensor(range(s)) / s).repeat(h, 1).flip(1)

    return dst_img
