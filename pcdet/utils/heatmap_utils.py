import torch
import numpy as np
import math


def gaussian_2d(shape, sigma=1, sigma2=None):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    if sigma2 is not None:
        sigma1 = sigma
        sigma2 = sigma2
        h = np.exp(-(x * x  / (2 * sigma1 * sigma1) + y * y / (2 * sigma2 * sigma2)))
    else:
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_2d_cylind(shape, center, resolution, scale, sigma=1, min_range_rho=0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map. [y, x]
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map. [rho, phi]
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    
    rho_res, theta_res = resolution[0], resolution[1]
    rho = (x + center[0]) * rho_res * scale + min_range_rho
    theta_offset = y * theta_res * scale
    # h = np.exp(-(rho_center * rho_center + (rho_center + rho_offset) ** 2 
    #     - 2 * rho_center * (rho_center + rho_offset) * np.cos(theta_offset) ) / (2 * sigma * sigma))
    
    rho_center = rho[:, (shape[1] - 1) // 2]

    distance_square = rho ** 2 + rho_center ** 2 - 2 * rho * rho_center * np.cos(theta_offset)

    h = np.exp(-(distance_square / ((rho_res * scale) ** 2) / (2 * sigma * sigma)))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1, y_factor=1, resolution=[0.05, 0.0021], scale=4):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked. shape (phi, rho)
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    try:
        y_radius = int(((radius * y_factor).floor().item()))
    except:
        y_radius = radius
    y_diameter = 2 * y_radius + 1
    # gaussian_center = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # index = np.linspace(0, diameter - 1, diameter, endpoint=True)
    # y_index = radius  / (y_radius + 1) * np.linspace(0, y_radius + 1, y_radius + 1, endpoint=True)
    # gaussian = np.zeros((y_diameter, diameter))

    # for i in range(gaussian_center.shape[0]):
    #     gaussian_half = np.interp(y_index, index, gaussian_center[i])
    #     gaussian[:, i] = np.concatenate((gaussian_half, gaussian_half[:-1][::-1]), axis=0)

    gaussian = gaussian_2d((y_diameter, diameter), sigma=diameter / 6, sigma2 = y_diameter / 6)

    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, y_radius), min(height - y, y_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[y_radius - top:y_radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    
    # gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # x, y = int(center[0]), int(center[1])
    
    # height, width = heatmap.shape[0:2]

    # left, right = min(x, radius), min(width - x, radius + 1)
    # top, bottom = min(y, radius), min(height - y, radius + 1)

    # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # masked_gaussian = torch.from_numpy(
    #     gaussian[radius - top:radius + bottom,
    #              radius - left:radius + right]).to(heatmap.device,
    #                                                torch.float32) 

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    # if radius != y_radius:
    #     print('radius: ', radius, ' y_radius: ', y_radius)
    #     print('y_factor: ', y_factor, ' y: ', y)
    #     import pdb
    #     pdb.set_trace()
    #     import cv2
    #     heatmap2=np.array(heatmap.cpu()) * 255
    #     heatmap2=heatmap2.astype(np.uint8)
    #     heatmap2=cv2.applyColorMap(heatmap2, cv2.COLORMAP_HOT)
    #     cv2.imshow('heatmap',heatmap2)
    #     cv2.waitKey(0)
    return heatmap



def draw_heatmap_gaussian_cylind(heatmap, center, radius, 
    k=1, y_factor=1, resolution=[0.05, 0.0021], scale=4, min_range_rho=0):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked. shape (phi, rho)
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    try:
        y_radius = int(((radius * y_factor).floor().item()))
    except:
        y_radius = radius
    y_diameter = 2 * y_radius + 1
    # gaussian_center = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # index = np.linspace(0, diameter - 1, diameter, endpoint=True)
    # y_index = radius  / (y_radius + 1) * np.linspace(0, y_radius + 1, y_radius + 1, endpoint=True)
    # gaussian = np.zeros((y_diameter, diameter))

    # for i in range(gaussian_center.shape[0]):
    #     gaussian_half = np.interp(y_index, index, gaussian_center[i])
    #     gaussian[:, i] = np.concatenate((gaussian_half, gaussian_half[:-1][::-1]), axis=0)

    # 2-dim gaussian distribution
    gaussian = gaussian_2d((y_diameter, diameter), sigma=diameter / 6, sigma2 = y_diameter / 6)

    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]
    
    # gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6).transpose(1, 0)

    # x, y = int(center[0]), int(center[1])
    
    # height, width = heatmap.shape[0:2]

    # left, right = min(x, radius), min(width - x, radius + 1)
    # top, bottom = min(y, radius), min(height - y, radius + 1)

    # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # masked_gaussian = torch.from_numpy(
    #     gaussian[radius - top:radius + bottom,
    #              radius - left:radius + right]).to(heatmap.device,
    #                                                torch.float32) 

    cy_radius_x = radius
    cy_radius_y = int(math.acos(1 - 0.5 * (radius / x) ** 2) / resolution[1] / scale)

    y_diameter = 2 * cy_radius_y + 1
    x_diameter = 2 * cy_radius_x + 1

    gaussian = gaussian_2d_cylind((y_diameter, x_diameter), sigma=(2 * radius + 1) / 6, 
            center=[x,y], resolution=resolution, scale=scale, min_range_rho=min_range_rho)

    left, right = min(x, cy_radius_x), min(width - x, cy_radius_x + 1)
    top, bottom = min(y, cy_radius_y), min(height - y, cy_radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[cy_radius_y - top:cy_radius_y + bottom,
                 cy_radius_x - left:cy_radius_x + right]).to(heatmap.device,
                                                   torch.float32)

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    # Visualization of heatmap
    # print('radius: {}, rho_radius: {}, theta_radius: {}'.format(radius, cy_radius_x, cy_radius_y))
    # print('gaussian shape: ', gaussian.shape)

    # import cv2
    # heatmap2=np.array(heatmap.cpu()) * 255
    # heatmap2=heatmap2.astype(np.uint8)
    # # heatmap2=cv2.applyColorMap(heatmap2, cv2.COLORMAP_HOT)
    # cv2.imshow('heatmap',heatmap2)
    # cv2.waitKey(0)

    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)