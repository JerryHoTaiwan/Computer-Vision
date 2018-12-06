import numpy as np
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    A = np.zeros((2*N, 8))
    # if you take solution 2:
    # A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    # TODO: compute H from A and b

    for i in range(N):
        b[2*i:2*(i+1)] = v[i].reshape(-1, 1)

    for i in range(N):
        A[2*i, 0:2] = u[i]#.reshape(-1, 1)
        A[2*i, 2:6] = np.array([1., 0., 0., 0.])
        A[2*i, 6] = - u[i, 0] * v[i, 0]
        A[2*i, 7] = - u[i, 1] * v[i, 0]

        A[2*i + 1, :3] = 0.
        A[2*i + 1, 3:5] = u[i]#.reshape(-1, 1)
        A[2*i + 1, 5] = 1.
        A[2*i + 1, 6] = - u[i, 0] * v[i, 1]
        A[2*i + 1, 7] = - u[i, 1] * v[i, 1]

    #A_inv = np.linalg.inv(A)
    H_onehot = np.linalg.solve(A, b)

    H[:2, :] = H_onehot[:6].reshape(2, 3)
    H[2, 0] = H_onehot[6]
    H[2, 1] = H_onehot[7]
    H[2, 2] = 1.

    return H

def back_warping(src_img, project, corner):
    h, w, ch = project.shape
    pro_corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    homography = solve_homography(pro_corners, corner)

    for y in range(h):
        for x in range(w):
            u_vec = np.array([[x, y, 1]])
            v_vec = np.dot(homography, u_vec.T)#.astype(np.int16)
            t_x = (v_vec[0] / v_vec[2])[0]
            t_y = (v_vec[1] / v_vec[2])[0]
            project[y, x] = bilinear(src_img, t_x, t_y)

def bilinear(src_img, t_x, t_y):
    x_left = round(t_x - int(t_x), 4)
    x_right = 1 - x_left
    y_low = round(t_y - int(t_y), 4)
    y_high = 1 - y_low

    int_x, int_y = int(round(t_x)), int(round(t_y))

    img_low_left = x_left * y_low * src_img[int_y, int_x]
    img_low_right = x_right * y_low * src_img[int_y, int_x + 1]
    img_high_left = x_left * y_high * src_img[int_y + 1, int_x]
    img_high_right = x_right * y_high * src_img[int_y + 1, int_x + 1]
    img_sum = img_low_left + img_low_right + img_high_left + img_high_right

    return img_sum

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic
    img_corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    homography = solve_homography(img_corners, corners)

    for y in range(h):
        for x in range(w):
            pixel = img[y, x, :]
            u_vec = np.array([[x, y, 1]])
            v_vec = np.dot(homography, u_vec.T)#.astype(np.int16)
            t_x = int(v_vec[0] / v_vec[2])
            t_y = int(v_vec[1] / v_vec[2])
            canvas[t_y, t_x, :] = pixel

def main():
    # Part 1
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/wu.jpg')
    img2 = cv2.imread('./input/ding.jpg')
    img3 = cv2.imread('./input/yao.jpg')
    img4 = cv2.imread('./input/kp.jpg')
    img5 = cv2.imread('./input/lee.jpg')

    corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]])
    corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
    corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
    corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
    corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])

    # TODO: some magic
    print ("========== PART 1 ===========")
    transform(img1, canvas, corners1)
    transform(img2, canvas, corners2)
    transform(img3, canvas, corners3)
    transform(img4, canvas, corners4)
    transform(img5, canvas, corners5)

    cv2.imwrite('part1.png', canvas)
    # Part 2
    print ("========== PART 2 ===========")

    img = cv2.imread('./input/screen.jpg')
    # TODO: some magic
    output2 = np.zeros((200, 200, 3))
    corner_QR = np.array([[1041, 370], [1100, 396], [984, 552], [1036, 599]])
    back_warping(img, output2, corner_QR)
    cv2.imwrite('part2.png', output2)

    # Part 3
    print ("========== PART 3 ===========")

    img_front = cv2.imread('./input/crosswalk_front.jpg')
    # TODO: some magic
    output3 = np.zeros((300, 500, 3))
    corners_crosswalk = np.array([[160, 129], [563, 129], [0, 286], [723, 286]])
    back_warping(img_front, output3, corners_crosswalk)

    cv2.imwrite('part3.png', output3)


if __name__ == '__main__':
    main()
