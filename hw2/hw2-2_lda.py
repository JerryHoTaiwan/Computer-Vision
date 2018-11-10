import numpy as np
import cv2
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier as KNC
from os import listdir, makedirs
from os.path import join, exists
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

class ImgData():

    def __init__(self, dataset_path='./hw2-2_data/'):

        self.dataset_path = dataset_path

        self.C = 40

        self.train_img = list()
        self.test_img = list()
        self.train_label = list()
        self.test_label = list()

        self.get_data()

    def get_data(self):

        for img_path in listdir(self.dataset_path):
            label, _index = img_path.split('_')
            index, _name = _index.split('.')
            img = cv2.imread(join(dataset_path, img_path), cv2.IMREAD_GRAYSCALE).flatten()
            if int(index) <= 7:
                self.train_img.append(img)
                self.train_label.append(label)
            else:
                self.test_img.append(img)
                self.test_label.append(label)

        self.train_img = np.array(self.train_img).astype(np.float64)
        self.train_label = np.array(self.train_label).astype(np.float64)
        self.test_img = np.array(self.test_img).astype(np.float64)
        self.test_label = np.array(self.test_label).astype(np.float64)


def PCA(X, k=4):

    #mean face
    X_mean = np.mean(X,axis=0).astype(np.uint8)
    X = X.T
    X_mean = X_mean.reshape(56*46,1)
    X = X - np.repeat(X_mean,[X.shape[1]],axis=1)
    #do pca
    print("start doing svd ...")
    U, s, V = np.linalg.svd(X, full_matrices=False)
    #print (X.shape,U.shape,s.shape,V.shape)
    s_sum = np.sum(s)
    #for i in range(k):
        #print ('ratio of w_' + str(i) + ': ', s[i] / s_sum)
    eigenfaces = U[:,:k]
    return X_mean, eigenfaces

def img2vec(img_set, mean, n, eigen):
    mean = mean.flatten()
    vec = np.zeros((img_set.shape[0],n))
    for i, img in enumerate(img_set):
        f_img = img.flatten() - mean
        for j in range(n):
            vec[i][j] = np.dot(f_img,eigen[:,j])    
    return vec

def LDA(vec_data, eigenfaces, N=280, C=40):

    # (40, samples, N-C)
    u_data = np.mean(vec_data, axis=1)
    u_all = np.mean(u_data, axis=0)
    S_b = np.dot((u_data - u_all).T, (u_data - u_all))

    # for numpy
    u_data = u_data.reshape(vec_data.shape[0], 1, vec_data.shape[2])

    S_w = np.zeros((N-C, N-C)).astype(np.float64)

    for i in range(C):
        for j in range(len(vec_data[i])):
            vec = vec_data[i][j]
            S_w += np.dot((vec - u_data[i]).T, (vec - u_data[i]))

    J_w = np.dot(np.linalg.inv(S_w), S_b)
    _lamda, w = np.linalg.eig(J_w)
    w = np.real(w)[:(C-1)].T
    eigen_NC = eigenfaces[:, :(N-C)]

    fisherface = np.dot(eigen_NC, w)
    return fisherface

def classified_data(data, label):
    samples = int(len(data) / 40)
    c_data = np.zeros((40, samples, data.shape[1])).astype(np.float64)
    count_data = np.zeros(40)

    for i, vec in enumerate(data):
        y = int(label[i]) - 1
        index = int(count_data[y])
        c_data[y, index, :] = vec
        count_data[y] += 1

    return c_data

def eigen2img(img):
        img = img * 1.0
        img -= np.min(img)
        img /= np.max(img)
        img = (img * 255).astype(np.uint8)
        return img

if __name__ == '__main__':

    dataset_path  = sys.argv[1]
    fisherface_path = sys.argv[2]

    img_data = ImgData(dataset_path)

    train_img = img_data.train_img
    train_label = img_data.train_label
    test_img = img_data.test_img
    test_label = img_data.test_label

    N = len(train_img)
    C = 40

    mean_face, eigenfaces = PCA(train_img, k=N-C)

    # projection

    train_vec = img2vec(train_img, mean_face, N-C, eigenfaces)
    cla_train_vec = classified_data(train_vec, train_label)
    print (cla_train_vec.shape, eigenfaces.shape)
    fisherfaces = LDA(cla_train_vec, eigenfaces)

    for i in range(5):
        save = eigen2img(fisherfaces.T[i].reshape(56,46))
        cv2.imwrite('eigenfaces/fisherfaces_'+str(i)+'.png', save)
