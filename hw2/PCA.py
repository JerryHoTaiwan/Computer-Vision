#from skimage import io
import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as KNC

w = 46
h = 56
num = 240

def accuracy(predict,truth):
    count = 0
    for i in range(predict.shape[0]):
        if (predict[i] == truth[i]):
            count += 1
    acc = count / predict.shape[0]
    return acc

def toimg(img):
        img = img * 1.0
        img -= np.min(img)
        img /= np.max(img)
        img = (img * 255).astype(np.uint8)
        return img

def get_allimg():
    X = np.zeros((240,56*46))
    Label = np.zeros((240,1))
    for i in range(1,41):
        for j in range(1,7):
            filename = 'hw2-2_data/'+str(i)+'_'+str(j)+'.png'  
            img = cv2.imread(filename,0).astype(np.uint8)
            new_img = img.flatten()
            print (filename)
            #new_img = transform.resize(img,(56,46), mode='reflect').flatten()
            index = (i-1) * 6 + j -1
            X[index] = new_img
            Label[index] = i
    return X,Label

def get_testimg():
    X = np.zeros((160,56*46))
    Label = np.zeros((160,1))
    for i in range(1,41):
        for j in range(7,11):
            filename = 'hw2-2_data/'+str(i)+'_'+str(j)+'.png'  
            img = cv2.imread(filename,0).astype(np.uint8)
            new_img = img.flatten()
            #new_img = transform.resize(img,(56,46), mode='reflect').flatten()
            index = (i-1) * 4 + j -7
            X[index] = new_img
            Label[index] = i
    return X,Label

def PCA(X, k=4):
        #print (X[:, 0])
        #mean face
        #global X_mean
        X_mean = np.mean(X,axis=0).astype(np.uint8)
        print (X_mean[:10])
        X = X.T
        print (X.shape,X_mean.shape)
        X_mean = X_mean.reshape(56*46,1)
        X = X - np.repeat(X_mean,[X.shape[1]],axis=1)
        #do pca
        print("start doing svd ...")
        U, s, V = np.linalg.svd(X, full_matrices=False)
        print (X.shape,U.shape,s.shape,V.shape)
        s_sum = np.sum(s)
        for i in range(k):
            print ('ratio of w_'+str(i)+': ',s[i]/s_sum)
        eigenfaces = U[:,:k]
        return eigenfaces

def reconstruct(img,k):
    global eigenfaces,img_mean
    eigen = np.copy(eigenfaces)
    f_img = img.flatten() - img_mean.flatten()
    weight = np.zeros(k)
    for i in range(k):
        weight[i] = np.dot(f_img,eigen[:,i])
    d_img = np.dot(weight,eigen.T).reshape(56*46,).astype(np.uint8)
    print ((d_img + X_mean).shape)
    re_img = (d_img + X_mean).reshape(56,46)
    return re_img

def img2vec(img,mean,n,eigen):
    vec = np.zeros((img.shape[0],n))
    for i in range(img.shape[0]):
        f_img = img[i].flatten() - mean
        for j in range(n):
            vec[i][j] = np.dot(f_img,eigen[:,j])    
    return vec

def K_NN(train,label,valid,truth,k,n):

    img_mean = np.mean(train,axis=0).astype(np.uint8)
    eigen = PCA(train,n)
    t_vec = img2vec(train,img_mean,n,eigen)
    v_vec = img2vec(valid,img_mean,n,eigen)

    print("doing k-nn ... ")
    neigh = KNC(n_neighbors=k)
    neigh.fit(t_vec,label)
    print (neigh.score(v_vec,truth))

    return 0

def MSE(predict,truth):
    size = predict.shape[0] * predict.shape[1]
    p = np.copy(predict).astype(np.float64)
    t = np.copy(truth).astype(np.float64)
    dev = np.abs(p-t).astype(np.float64)
    dev_2 = np.multiply(dev,dev).astype(np.float64)
    print (np.amax(t),np.amax(p),np.amax(dev))
    mse = np.sum(dev_2) / size
    return mse

def dev_3(data):
    size_0 = int(data.shape[0] / 3)
    size_1 = data.shape[1]
    data_0 = np.zeros((size_0,size_1))
    data_1 = np.zeros((size_0,size_1))
    data_2 = np.zeros((size_0,size_1))

    for i in range(data.shape[0]):
        index = int(i / 3)
        if (i % 3 == 0):
            data_0[index] = data[i]
        elif (i % 3 == 1):
            data_1[index] = data[i]
        elif (i % 3 == 2):
            data_2[index] = data[i]
    return (data_0,data_1,data_2)

######P1######

(X,Label) = get_allimg()

X_mean = np.mean(X,axis=0).astype(np.uint8)
img_mean = X_mean.reshape(56,46).astype(np.float64)

#cv2.imwrite('average_2.png',X_mean.reshape(56,46))

n = int(sys.argv[1])
k = int(sys.argv[2])

eigenfaces = PCA(X, n)

print (eigenfaces.shape)
for i in range(k):
    save = toimg(eigenfaces.T[i].reshape(56,46))
    cv2.imwrite('eigenfaces/eigenfaces_'+str(i)+'.png',save)

#####P2#####

target = cv2.imread('hw2-2_data/1_1.png',0).astype(np.uint8)
re_target = reconstruct(target,n)
cv2.imwrite('reconstruction/re_'+str(n)+'.png',re_target)

print ('The MSE is: ' + str(MSE(re_target,target)))

####P3####

(X_0,X_1,X_2) = dev_3(X)
(L_0,L_1,L_2) = dev_3(Label)

Train_0 = np.concatenate((X_1,X_2),axis=0)
Train_1 = np.concatenate((X_0,X_2),axis=0)
Train_2 = np.concatenate((X_0,X_1),axis=0)

Label_0 = np.concatenate((L_1,L_2),axis=0)
Label_1 = np.concatenate((L_0,L_2),axis=0)
Label_2 = np.concatenate((L_0,L_1),axis=0)

#K_NN(Train_0,Label_0,X_0,L_0,k,n)
#K_NN(Train_1,Label_1,X_1,L_1,k,n)
#K_NN(Train_2,Label_2,X_2,L_2,k,n)
(Test,Truth) = get_testimg()
K_NN(X,Label,Test,Truth,k,n)
