import numpy as np
import cv2
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier as KNC
from os import listdir, makedirs
from os.path import join, exists
from sklearn.manifold import TSNE
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
"""
class ImgData():

    def __init__(self, dataset_path='./hw2-2_data/'):

        self.dataset_path = dataset_path

        self.train_img = list()
        self.test_img = list()
        self.train_label = list()
        self.test_label = list()

        self.valid_img_0 = list()
        self.valid_img_1 = list()
        self.valid_img_2 = list()

        self.valid_label_0 = list()
        self.valid_label_1 = list()
        self.valid_label_2 = list()

        self.get_data()
        print (self.train_img.shape)

    def get_data(self):

        for img_path in listdir(self.dataset_path):
            label, _index = img_path.split('_')
            index, _name = _index.split('.')
            img = cv2.imread(join(dataset_path, img_path), cv2.IMREAD_GRAYSCALE).flatten()
            if int(index) <= 7:
                self.train_img.append(img)
                self.train_label.append(label)
                
                if int(index) <= 2:
                    self.valid_img_0.append(img)
                    self.valid_label_0.append(label)

                elif int(index) <= 4:
                    self.valid_img_1.append(img)
                    self.valid_label_1.append(label)
                else:
                    self.valid_img_2.append(img)
                    self.valid_label_2.append(label)
            else:
                self.test_img.append(img)
                self.test_label.append(label)

        self.train_img = np.array(self.train_img).astype(np.float64)
        self.train_label = np.array(self.train_label).astype(np.float64)
        self.test_img = np.array(self.test_img).astype(np.float64)
        self.test_label = np.array(self.test_label).astype(np.float64)

        self.valid_img_0 = np.array(self.valid_img_0).astype(np.float64)
        self.valid_img_1 = np.array(self.valid_img_1).astype(np.float64)
        self.valid_img_2 = np.array(self.valid_img_2).astype(np.float64)

        self.valid_label_0 = np.array(self.valid_label_0).astype(np.float64)
        self.valid_label_1 = np.array(self.valid_label_1).astype(np.float64)
        self.valid_label_2 = np.array(self.valid_label_2).astype(np.float64)

def PCA(X, k=4):

    #mean face
    X_mean = np.mean(X,axis=0).astype(np.uint8)
    X = X.T
    X_mean = X_mean.reshape(56*46,1)
    X = X - np.repeat(X_mean,[X.shape[1]],axis=1)
    #do pca
    #print("start doing svd ...")
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

def LDA(vec_data, eigenfaces, N=280):

    # (40, samples, N-C)
    classes, samples, dim = vec_data.shape

    S_w = np.zeros((dim, dim)).astype(np.float64)
    S_b = np.zeros((dim, dim)).astype(np.float64)

    for i in range(classes):
        for j in range(samples):
            u_i = np.mean(vec_data[i, :], axis=0)
            sw_i = vec_data[i, j, :].reshape(-1, 1) - u_i.reshape(-1, 1)
            sw_i = np.multiply(sw_i, sw_i.T)
            S_w += sw_i

    u_all = np.mean(vec_data.reshape(-1, dim),axis=0)

    for i in range(classes):
        u_i = np.mean(vec_data[i, :], axis=0)
        sb_i = u_i.reshape(-1, 1) - u_all.reshape(-1, 1)
        sb_i = np.multiply(sb_i, sb_i.T)
        S_b += sb_i

    inv_sw = np.linalg.inv(S_w)
    J_w = np.dot(inv_sw, S_b)
    _lamda, w = np.linalg.eig(J_w)

    w = np.real(w)[:, :(classes-1)]
    eigen_NC = eigenfaces[:, :(N-classes)]

    fisherface = np.dot(eigen_NC, w)
    return fisherface, w

def K_NN(train, t_label, valid, v_label, k, n):

    N = len(train)
    C = 40

    img_mean = np.mean(train,axis=0).astype(np.uint8)
    _mean, eigen = PCA(train, N-C)
    t_vec_pca = img2vec(train, img_mean, N-C, eigen)
    v_vec_pca = img2vec(valid, img_mean, N-C, eigen)

    t_vec_pca_cla = classified_data(t_vec_pca, t_label)
    _fisherface, w = LDA(t_vec_pca_cla, eigen, N)

    w = w[:, :n]

    t_vec_lda = np.dot(t_vec_pca, w)
    v_vec_lda = np.dot(v_vec_pca, w)

    #print("doing k-nn ... ")
    neigh = KNC(n_neighbors=k)
    neigh.fit(t_vec_lda, t_label)
    print (neigh.score(v_vec_lda, v_label))

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

def plot_embedding(data, label):
    x = data[:, 0]
    y = data[:, 1]
    c = label
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(x, y, c=c, cmap=cm.tab20c, linewidth=0, antialiased=False)
    fig.colorbar(scat, shrink=0.5, aspect=5)
    ax.set_title('TSNE')
    plt.savefig('lda_TSNE.png')

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
    fisherfaces, w = LDA(cla_train_vec, eigenfaces)

    first_ff = eigen2img(fisherfaces.T[0].reshape(56,46))
    cv2.imwrite(fisherface_path, first_ff)

    """

    for i in range(5):
        save = eigen2img(fisherfaces.T[i].reshape(56,46))
        cv2.imwrite('eigenfaces/fisherfaces_'+str(i)+'.png', save)

    test_vec_pca = img2vec(test_img, mean_face, N-C, eigenfaces)
    test_vec_lda = np.dot(test_vec_pca, w)
    test_embedded = TSNE(n_components=2).fit_transform(test_vec_lda)
    plot_embedding(test_embedded, test_label)

    # cross-validation
    valid_img_0 = img_data.valid_img_0
    valid_img_1 = img_data.valid_img_1
    valid_img_2 = img_data.valid_img_2

    valid_label_0 = img_data.valid_label_0
    valid_label_1 = img_data.valid_label_1
    valid_label_2 = img_data.valid_label_2

    for n in [3, 10, 39]:
        for k in [1, 3, 5]:
            
            print ("n: ", n, "k: ", k)

            train_img_0 = np.concatenate((valid_img_1, valid_img_2), axis=0)
            train_img_1 = np.concatenate((valid_img_0, valid_img_2), axis=0)
            train_img_2 = np.concatenate((valid_img_0, valid_img_1), axis=0)

            train_label_0 = np.concatenate((valid_label_1, valid_label_2), axis=0)
            train_label_1 = np.concatenate((valid_label_0, valid_label_2), axis=0)
            train_label_2 = np.concatenate((valid_label_0, valid_label_1), axis=0)

            K_NN(train_img_0, train_label_0, valid_img_0, valid_label_0, k, n)
            K_NN(train_img_1, train_label_1, valid_img_1, valid_label_1, k, n)
            K_NN(train_img_2, train_label_2, valid_img_2, valid_label_2, k, n)

            print ("Testing")
            K_NN(train_img, train_label, test_img, test_label, k, n)
    """