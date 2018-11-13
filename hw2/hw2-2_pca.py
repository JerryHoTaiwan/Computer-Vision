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
    s_sum = np.sum(s)
    #for i in range(k):
        #print ('ratio of w_' + str(i) + ': ', s[i] / s_sum)
    eigenfaces = U[:,:k]
    return X_mean, eigenfaces

def eigen2img(img):
        img = img * 1.0
        img -= np.min(img)
        img /= np.max(img)
        img = (img * 255).astype(np.uint8)
        return img

def reconstruct(img, k, eigenfaces, img_mean):

    X_mean = img_mean.flatten()
    eigen = np.copy(eigenfaces)
    f_img = img.flatten() - img_mean.flatten()
    weight = np.zeros(k)
    for i in range(k):
        weight[i] = np.dot(f_img,eigen[:, i])
    d_img = np.dot(weight, eigen.T[:k]).reshape(56*46,).astype(np.float64)
    re_img = (d_img + X_mean).reshape(56,46)
    return re_img

def MSE(predict, truth):
    size = predict.shape[0] * predict.shape[1]
    p = np.copy(predict).astype(np.float64)
    t = np.copy(truth).astype(np.float64)
    dev = np.abs(p-t).astype(np.float64)
    dev_2 = np.multiply(dev,dev).astype(np.float64)
    #print (np.amax(t),np.amax(p),np.amax(dev))
    mse = np.sum(dev_2) / size
    return mse

def img2vec(img_set, mean, n, eigen):
    mean = mean.flatten()
    vec = np.zeros((img_set.shape[0],n))
    for i, img in enumerate(img_set):
        f_img = img.flatten() - mean
        #print (f_img.shape, eigen.shape)
        for j in range(n):
            vec[i, j] = np.dot(f_img,eigen[:,j])    
    return vec

def plot_embedding(data, label):
    x = data[:, 0]
    y = data[:, 1]
    c = label
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(x, y, c=c, cmap=cm.tab20c, linewidth=0, antialiased=False)
    fig.colorbar(scat, shrink=0.5, aspect=5)
    ax.set_title('TSNE')
    plt.savefig('pca_TSNE.png')

def K_NN(train, t_label, valid, v_label, k, n):

    img_mean = np.mean(train,axis=0).astype(np.uint8)
    _mean, eigen = PCA(train,n)
    t_vec = img2vec(train, img_mean, n, eigen)
    v_vec = img2vec(valid, img_mean, n, eigen)

    print("doing k-nn ... ")
    neigh = KNC(n_neighbors=k)
    neigh.fit(t_vec, t_label)
    print (neigh.score(v_vec, v_label))

    return 0

if __name__ == '__main__':

    dataset_path  = sys.argv[1]
    test_img_path = sys.argv[2]
    rec_img_path  = sys.argv[3]

    img_data = ImgData(dataset_path)

    train_img = img_data.train_img
    train_label = img_data.train_label
    test_img = img_data.test_img
    test_label = img_data.test_label

    mean_face, eigenfaces = PCA(train_img, k=len(train_img)-1)

    # For submission
    target_img = cv2.imread(test_img_path,0).astype(np.float64)
    re_target = reconstruct(target_img, len(train_img)-1, eigenfaces, mean_face)
    cv2.imwrite(rec_img_path, re_target)

    """
    # eigenface and meanface

    cv2.imwrite('meanface.png', mean_face.reshape(56, 46))
    
    if not exists('./eigenfaces'):
        makedirs('./eigenfaces/')
    for i in range(5):
        save = eigen2img(eigenfaces.T[i].reshape(56,46))
        cv2.imwrite('eigenfaces/eigenfaces_'+str(i)+'.png', save)

    # reconstruction
    
    target_img = cv2.imread(test_img_path,0).astype(np.float64)


    for n in [5, 50, 150, len(train_img)-1]:
        re_target = reconstruct(target_img, n, eigenfaces, mean_face)
        cv2.imwrite('reconstruction/re_'+str(n)+'.png', re_target)
        print ("MSE: ", MSE(re_target, target_img))

    # TSNE

    test_vec = img2vec(test_img, mean_face, 100, eigenfaces)
    test_embedded = TSNE(n_components=2).fit_transform(test_vec)
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