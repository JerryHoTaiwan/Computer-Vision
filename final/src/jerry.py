import numpy as np
import cv2

def test_range(left_img, right_img): # real: whether it is a real image pair (involve negative disparity) min/max _dis = the estimated range of disparity
    surf = cv2.xfeatures2d.SURF_create(1000)
    bf = cv2.BFMatcher()

    left_kp, left_des = surf.detectAndCompute(left_img,None)
    right_kp, right_des = surf.detectAndCompute(right_img,None)
    matches = bf.knnMatch(left_des, right_des, k=2)
        
    good = list()
    pos = 0
    for (m, n) in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    dis = list()
    pos = 0
    neg = 0
    for _m in good:
        left_idx = left_kp[_m.queryIdx].pt
        right_idx = right_kp[_m.trainIdx].pt
        #print (left_idx, right_idx)
        dis.append(left_idx[0] - right_idx[0])

        if (left_idx[0] > right_idx[0]):
            pos += 1
        else:
            neg += 1
    dis = (np.array(dis))
    dis = np.sort(dis)
    #print (dis)
    real = (neg / (pos + neg)) < 0.9
    min_dis = 0
    max_dis = 50

    for i in range(len(dis)):
        if dis[i] > -60 and dis[i+1] - dis[i] < 2 and dis[i+2] - dis[i] < 2:
            min_dis = dis[i]
            break

    for i in range(len(dis)):
        if dis[-1-i] < 60 and dis[-1-i] - dis[-1-(i+1)] < 2 and dis[-1-i] - dis[-1-(i+2)] < 2:
            max_dis = dis[-1-i]
            break

    max_dis += 1
    min_dis -= 1

    return real, int(min_dis), int(max_dis)

def bilateralfilter_float(image, texture, sigma_s, sigma_r):
    r = int(np.ceil(3 * sigma_s))
    # Image padding
    if image.ndim == 3:
        h, w, ch = image.shape
        I = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
    elif image.ndim == 2:
        h, w = image.shape
        I = np.pad(image, ((r, r), (r, r)), 'symmetric').astype(np.float32)
    else:
        print('Input image is not valid!')
        return image
    # Check texture size and do padding
    if texture.ndim == 3:
        ht, wt, cht = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)
    elif texture.ndim == 2:
        ht, wt = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
    # Pre-compute
    output = np.zeros_like(image)
    scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
    scaleFactor_r = 1 / (2 * sigma_r * sigma_r)
    # A lookup table for range kernel
    LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
    # Generate a spatial Gaussian function
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
    # Main body
    if I.ndim == 2 and T.ndim == 2:     # I1T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[np.abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    elif I.ndim == 3 and T.ndim == 2:     # I3T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 3 and T.ndim == 3:     # I3T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 2 and T.ndim == 3:     # I1T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    else:
        print('Something wrong!')
        return image

    # return np.clip(output, 0, 255)
    return output