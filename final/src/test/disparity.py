import numpy as np
import cv2

L1 = 34
L2 = 17
tao1 = 20
tao2 = 6

G1 = 1.0
G2 = 3.0
tao_so = 15
tao_s = 1
tao_h = 0.05

class disp_calculator:

    def __init__(self, imgL, imgR):
        self.imgL = self.high_contrast(imgL,25,45,4).astype('int')
        self.imgR = self.high_contrast(imgR,13,40,2).astype('int')
        self.h, self.w, self.ch = imgL.shape

    def high_contrast(self,img,space,sigma_c,sigma_s):
        I = np.zeros(img.shape)
        I[:,:,0] = cv2.equalizeHist(img[:,:,0])
        I[:,:,1] = cv2.equalizeHist(img[:,:,1])
        I[:,:,2] = cv2.equalizeHist(img[:,:,2])
        I = cv2.bilateralFilter(I.astype('uint8'),space,sigma_c,sigma_s,borderType=cv2.BORDER_REFLECT)
        return I

    '''
    def find_disp_range(self):
        self.is_real, self.neg_disp, self.pos_disp = test_range(self.imgL,self.imgR)
    '''
    def set_max_disparity(self,disp):
        self.max_disp = int(disp)

    def find_L_disparity(self):
        self.u_arms_L = self.arm_union_L()
        C_AD = self.intense_cost_L()
        C_C = self.census_cost_L(11,11)
        costs = self.cost_merge(C_AD,C_C)
        cost_aggr = self.aggregate_cost_H(self.u_arms_L,costs)
        cost_aggr = self.aggregate_cost_V(self.u_arms_L,cost_aggr)
        cost_aggr = self.aggregate_cost_H(self.u_arms_L,cost_aggr)
        cost_aggr = self.aggregate_cost_V(self.u_arms_L,cost_aggr)
        cost_aggr = self.scanline_optimize_L(cost_aggr)
        return cost_aggr

    def find_R_disparity(self):
        self.u_arms_R = self.arm_union_R()
        C_AD = self.intense_cost_R()
        C_C = self.census_cost_R(11,11)
        costs = self.cost_merge(C_AD,C_C)
        cost_aggr = self.aggregate_cost_H(self.u_arms_R,costs)
        cost_aggr = self.aggregate_cost_V(self.u_arms_R,cost_aggr)
        cost_aggr = self.aggregate_cost_H(self.u_arms_R,cost_aggr)
        cost_aggr = self.aggregate_cost_V(self.u_arms_R,cost_aggr)
        cost_aggr = self.scanline_optimize_R(cost_aggr)
        return cost_aggr

    def set_arms(self):
        self.armsL = arm_compute(self.imgL)
        self.armsR = arm_compute(self.imgR)

    def arm_union_L(self):
        arms = np.zeros((self.h,self.w,self.max_disp,4))
        for d in range(self.max_disp):
            armsR_shift = np.concatenate( ( np.repeat(self.armsR[:,0,:].reshape(self.h,1,4), d, axis=1  ), self.armsR[:,:self.w-d,:] ),axis=1)
            arms[:,:,d,:] = np.minimum(self.armsL,armsR_shift)
        return arms.astype("int")

    def arm_union_R(self):
        arms = np.zeros((self.h,self.w,self.max_disp,4))
        for d in range(self.max_disp):
            armsL_shift = np.concatenate( ( self.armsL[:,d:,:], (np.repeat(self.armsL[:,-1,:].reshape(self.h,1,4), d, axis=1) ) ),axis=1)
            arms[:,:,d,:] = np.minimum(self.armsR,armsL_shift)
        return arms.astype("int")

    def intense_cost_L(self):
        e_d = np.zeros((self.h,self.w,self.max_disp))
        for d in range(self.max_disp):
            imgR_shift = np.concatenate( ( np.repeat(self.imgR[:,0,:].reshape(self.h,1,3), d, axis=1  ), self.imgR[:,:self.w-d,:] ),axis=1)
            e_d[:,:,d] = np.sum(np.abs(self.imgL - imgR_shift),axis=2)
        return e_d

    def intense_cost_R(self):
        e_d = np.zeros((self.h,self.w,self.max_disp))
        for d in range(self.max_disp):
            imgL_shift = np.concatenate( (self.imgL[:,d:,:], ( np.repeat(self.imgL[:,-1,:].reshape(self.h,1,3), d, axis=1  )) ),axis=1)
            e_d[:,:,d] = np.sum(np.abs(self.imgR - imgL_shift),axis=2)
        return e_d

    def census_cost_L(self,ww,wl):
        area = (2*wl+1)*(2*ww+1)
        e_d = np.zeros((self.h,self.w,self.max_disp))
        for d in range(self.max_disp):
            censusR_shift = np.concatenate( ( np.repeat(self.census_array_R[:,0,:].reshape(self.h,1,area), d, axis=1  ), self.census_array_R[:,:self.w-d,:] ),axis=1)
            e_d[:,:,d] = np.sum((np.logical_xor(self.census_array_L,censusR_shift)),axis=2)
    
        return e_d

    def census_cost_R(self,ww,wl):
        area = (2*wl+1)*(2*ww+1)
        e_d = np.zeros((self.h,self.w,self.max_disp))
        for d in range(self.max_disp):
            censusL_shift = np.concatenate( (self.census_array_L[:,d:,:], ( np.repeat(self.census_array_L[:,-1,:].reshape(self.h,1,area), d, axis=1 ) ) ),axis=1)
            e_d[:,:,d] = np.sum((np.logical_xor(self.census_array_R,censusL_shift)),axis=2)
    
        return e_d

    def build_census_window(self,ww,wl):
        area = (2*wl+1)*(2*ww+1)
        
        meanL = np.mean(self.imgL,axis=2)
        meanR = np.mean(self.imgR,axis=2)
        meanL_pad = cv2.copyMakeBorder(meanL,wl,wl,ww,ww,cv2.BORDER_REPLICATE)
        meanR_pad = cv2.copyMakeBorder(meanR,wl,wl,ww,ww,cv2.BORDER_REPLICATE)
    
        census_array_L = np.zeros((self.h,self.w,area))
        census_array_R = np.zeros((self.h,self.w,area))
        for x in range(ww,self.w+ww):
            for y in range(wl,self.h+wl):
                _x = x - ww
                _y = y - wl
                winL = meanL_pad[y-wl:y+wl+1,x-ww:x+ww+1].reshape(area)
                winR = meanR_pad[y-wl:y+wl+1,x-ww:x+ww+1].reshape(area)
                census_array_L[_y,_x,:] = np.array(winL >= winL[int((area+1)/2)])
                census_array_R[_y,_x,:] = np.array(winR >= winR[int((area+1)/2)])
        self.census_array_L = census_array_L
        self.census_array_R = census_array_R


    def cost_merge(self,C_AD,C_C):
        ld_AD = 20*3
        ld_C = 350
        costs = (1-np.exp(-C_C/ld_C))+(1-np.exp(-C_AD/ld_AD))
        return costs

    def aggregate_cost_H(self,arms,costs):
        H_num = np.ones((self.h,self.w,self.max_disp))*3
        U_num = np.ones((self.h,self.w,self.max_disp))*9
        # step 1, compute horizontal integral image
        S_H = np.zeros((self.h,self.w,self.max_disp))
        for x in range(1,self.w):
            S_H[:,x,:] = S_H[:,x-1,:] + costs[:,x,:]
        # step 2, compute horizontal aggregation cost
        S_H = np.concatenate((np.zeros(self.w*self.max_disp).reshape(1,self.w,self.max_disp),S_H),axis=0)
        S_H = np.concatenate((np.zeros((self.h+1)*self.max_disp).reshape(self.h+1,1,self.max_disp),S_H),axis=1)
        E_H = np.zeros((self.h,self.w,self.max_disp))
        for x in range(1,self.w-1):
            for d in range(self.max_disp):
                E_H[:,x,d] = S_H[np.arange(self.h)+1,x+arms[:,x,d,0]+1,d] - S_H[np.arange(self.h)+1,x-arms[:,x,d,1],d]
                H_num[:,x,d] = arms[:,x,d,0] + arms[:,x,d,1] + 1
        # step 3, compute vertical interal image
        H_num_V = np.copy(H_num)
        S_V = np.zeros((self.h,self.w,self.max_disp))
        for y in range(1,self.h):
            S_V[y,:,:] = S_V[y-1,:,:] + E_H[y,:,:]
            H_num_V[y,:,:] = H_num_V[y-1,:,:] + H_num[y,:,:]
        # step 4, compute full aggregation matching cost
        S_V = np.concatenate((np.zeros(self.w*self.max_disp).reshape(1,self.w,self.max_disp),S_V),axis=0)
        S_V = np.concatenate((np.zeros((self.h+1)*self.max_disp).reshape(self.h+1,1,self.max_disp),S_V),axis=1)
        H_num_V = np.concatenate((np.zeros(self.w*self.max_disp).reshape(1,self.w,self.max_disp),H_num_V),axis=0)
        H_num_V = np.concatenate((np.zeros((self.h+1)*self.max_disp).reshape(self.h+1,1,self.max_disp),H_num_V),axis=1)
        E = np.zeros((self.h,self.w,self.max_disp))
        for y in range(1,self.h-1):
            for d in range(self.max_disp):
                E[y,:,d] = S_V[y+arms[y,:,d,2]+1,np.arange(self.w)+1,d] - S_V[y-arms[y,:,d,3],np.arange(self.w)+1,d]
                U_num[y,:,d] = H_num_V[y+arms[y,:,d,2]+1,np.arange(self.w)+1,d] - H_num_V[y-arms[y,:,d,3],np.arange(self.w)+1,d]
        E /= U_num
        return E

    def aggregate_cost_V(self,arms,costs):
        V_num = np.ones((self.h,self.w,self.max_disp))*3
        U_num = np.ones((self.h,self.w,self.max_disp))*9
        # step 1, compute horizontal integral image
        S_V = np.zeros((self.h,self.w,self.max_disp))
        for y in range(1,self.h):
            S_V[y,:,:] = S_V[y-1,:,:] + costs[y,:,:]
        # step 2, compute horizontal aggregation cost
        S_V = np.concatenate((np.zeros(self.w*self.max_disp).reshape(1,self.w,self.max_disp),S_V),axis=0)
        S_V = np.concatenate((np.zeros((self.h+1)*self.max_disp).reshape(self.h+1,1,self.max_disp),S_V),axis=1)
        E_V = np.zeros((self.h,self.w,self.max_disp))
        for y in range(1,self.h-1):
            for d in range(self.max_disp):
                E_V[y,:,d] = S_V[y+arms[y,:,d,2]+1,np.arange(self.w)+1,d] - S_V[y-arms[y,:,d,3],np.arange(self.w)+1,d]
                V_num[y,:,d] = arms[y,:,d,2] + arms[y,:,d,3] + 1
        # step 3, compute vertical interal image
        V_num_H = np.copy(V_num)
        S_H = np.zeros((self.h,self.w,self.max_disp))
        for x in range(1,self.w):
            S_H[:,x,:] = S_H[:,x-1,:] + E_V[:,x,:]
            V_num_H[:,x,:] = V_num_H[:,x-1,:] + V_num[:,x,:]
        # step 4, compute full aggregation matching cost
        S_H = np.concatenate((np.zeros(self.w*self.max_disp).reshape(1,self.w,self.max_disp),S_H),axis=0)
        S_H = np.concatenate((np.zeros((self.h+1)*self.max_disp).reshape(self.h+1,1,self.max_disp),S_H),axis=1)
        V_num_H = np.concatenate((np.zeros(self.w*self.max_disp).reshape(1,self.w,self.max_disp),V_num_H),axis=0)
        V_num_H = np.concatenate((np.zeros((self.h+1)*self.max_disp).reshape(self.h+1,1,self.max_disp),V_num_H),axis=1)
        E = np.zeros((self.h,self.w,self.max_disp))
        for x in range(1,self.w-1):
            for d in range(self.max_disp):
                E[:,x,d] = S_H[np.arange(self.h)+1,x+arms[:,x,d,0]+1,d] - S_H[np.arange(self.h)+1,x-arms[:,x,d,1],d]
                U_num[:,x,d] = V_num_H[np.arange(self.h)+1,x+arms[:,x,d,0]+1,d] - V_num_H[np.arange(self.h)+1,x-arms[:,x,d,1],d]
        E /= U_num
        return E

    def scanline_optimize_L(self,costs):
        imgR_shift = np.zeros((self.max_disp,self.h,self.w,self.ch))
        for d in range(self.max_disp):
            imgR_shift[d] = np.concatenate( ( np.repeat(self.imgR[:,0,:].reshape(self.h,1,3), d, axis=1  ), self.imgR[:,:self.w-d,:] ),axis=1)
        C_r = np.repeat(costs.reshape(self.h,self.w,self.max_disp,1),4,axis=3)
        for x in range(1,self.w):
            for d in range(self.max_disp):
                Q = C_r[:,x-1,d,0]
                R = C_r[:,x-1,d+1,0] if d+1 < self.max_disp else np.ones(self.h)*10000000
                S = C_r[:,x-1,d-1,0] if d-1 > 0 else np.ones(self.h)*10000000
                T = np.min(C_r[:,x-1,:,0],axis=1)
                P1, P2 = penalty(self.imgL[:,x,:],self.imgL[:,x-1,:],imgR_shift[d,:,x,:],imgR_shift[d,:,x-1,:])
                C_r[:,x,d,0] = costs[:,x,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        for x in range(self.w-2,0,-1):
            for d in range(self.max_disp):
                Q = C_r[:,x+1,d,1]
                R = C_r[:,x+1,d+1,1] if d+1 < self.max_disp else np.ones(self.h)*10000000
                S = C_r[:,x+1,d-1,1] if d-1 > 0 else np.ones(self.h)*10000000
                T = np.min(C_r[:,x+1,:,1],axis=1)
                P1, P2 = penalty(self.imgL[:,x,:],self.imgL[:,x+1,:],imgR_shift[d,:,x,:],imgR_shift[d,:,x+1,:])
                C_r[:,x,d,1] = costs[:,x,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        for y in range(1,self.h):
            for d in range(self.max_disp):
                Q = C_r[y-1,:,d,2]
                R = C_r[y-1,:,d+1,2] if d+1 < self.max_disp else np.ones(self.w)*10000000
                S = C_r[y-1,:,d-1,2] if d-1 > 0 else np.ones(self.w)*10000000
                T = np.min(C_r[y-1,:,:,2],axis=1)
                P1, P2 = penalty(self.imgL[y,:,:],self.imgL[y-1,:,:],imgR_shift[d,y,:,:],imgR_shift[d,y-1,:,:])
                C_r[y,:,d,2] = costs[y,:,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        for y in range(self.h-2,0,-1):
            for d in range(self.max_disp):
                Q = C_r[y+1,:,d,3]
                R = C_r[y+1,:,d+1,3] if d+1 < self.max_disp else np.ones(self.w)*10000000
                S = C_r[y+1,:,d-1,3] if d-1 > 0 else np.ones(self.w)*10000000
                T = np.min(C_r[y+1,:,:,3],axis=1)
                P1, P2 = penalty(self.imgL[y,:,:],self.imgL[y+1,:,:],imgR_shift[d,y,:,:],imgR_shift[d,y+1,:,:])
                C_r[y,:,d,3] = costs[y,:,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        return np.mean(C_r,axis=3)

    def scanline_optimize_R(self,costs):
        imgL_shift = np.zeros((self.max_disp,self.h,self.w,self.ch))
        for d in range(self.max_disp):
            imgL_shift[d] = np.concatenate( (self.imgL[:,d:,:], ( np.repeat(self.imgL[:,0,:].reshape(self.h,1,3), d, axis=1 ) ) ),axis=1)
        C_r = np.repeat(costs.reshape(self.h,self.w,self.max_disp,1),4,axis=3)
        for x in range(1,self.w):
            for d in range(self.max_disp):
                Q = C_r[:,x-1,d,0]
                R = C_r[:,x-1,d+1,0] if d+1 < self.max_disp else np.ones(self.h)*10000000
                S = C_r[:,x-1,d-1,0] if d-1 > 0 else np.ones(self.h)*10000000
                T = np.min(C_r[:,x-1,:,0],axis=1)
                P1, P2 = penalty_2(self.imgR[:,x,:],self.imgR[:,x-1,:],imgL_shift[d,:,x,:],imgL_shift[d,:,x-1,:])
                C_r[:,x,d,0] = costs[:,x,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        for x in range(self.w-2,0,-1):
            for d in range(self.max_disp):
                Q = C_r[:,x+1,d,1]
                R = C_r[:,x+1,d+1,1] if d+1 < self.max_disp else np.ones(self.h)*10000000
                S = C_r[:,x+1,d-1,1] if d-1 > 0 else np.ones(self.h)*10000000
                T = np.min(C_r[:,x+1,:,1],axis=1)
                P1, P2 = penalty_2(self.imgR[:,x,:],self.imgR[:,x+1,:],imgL_shift[d,:,x,:],imgL_shift[d,:,x+1,:])
                C_r[:,x,d,1] = costs[:,x,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        for y in range(1,self.h):
            for d in range(self.max_disp):
                Q = C_r[y-1,:,d,2]
                R = C_r[y-1,:,d+1,2] if d+1 < self.max_disp else np.ones(self.w)*10000000
                S = C_r[y-1,:,d-1,2] if d-1 > 0 else np.ones(self.w)*10000000
                T = np.min(C_r[y-1,:,:,2],axis=1)
                P1, P2 = penalty_2(self.imgR[y,:,:],self.imgR[y-1,:,:],imgL_shift[d,y,:,:],imgL_shift[d,y-1,:,:])
                C_r[y,:,d,2] = costs[y,:,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        for y in range(self.h-2,0,-1):
            for d in range(self.max_disp):
                Q = C_r[y+1,:,d,3]
                R = C_r[y+1,:,d+1,3] if d+1 < self.max_disp else np.ones(self.w)*10000000
                S = C_r[y+1,:,d-1,3] if d-1 > 0 else np.ones(self.w)*10000000
                T = np.min(C_r[y+1,:,:,3],axis=1)
                P1, P2 = penalty_2(self.imgR[y,:,:],self.imgR[y+1,:,:],imgL_shift[d,y,:,:],imgL_shift[d,y+1,:,:])
                C_r[y,:,d,3] = costs[y,:,d] + np.minimum(np.minimum(Q,T+P2),np.minimum(R,S)+P1) - T
        return np.mean(C_r,axis=3)

    def region_vote(self,D_L,outlier,d_max):
        h,w = D_L.shape
        vote = np.zeros((h,w,d_max))
    
        category = np.zeros((h,w,d_max)) #convert to categorical form
        for d in range(d_max):
                category[np.logical_and(D_L == d, outlier == 1),d] = 1
    
        for d in range(d_max):
            pool = category[:,:,d]
            S_H = np.zeros((h,w))
            for x in range(1,w):
                S_H[:,x] = S_H[:,x-1] + pool[:,x]
            S_H = np.concatenate((np.zeros(w).reshape(1,w),S_H),axis=0)
            S_H = np.concatenate((np.zeros(h+1).reshape(h+1,1),S_H),axis=1)
            E_H = np.zeros((h,w))
            for x in range(1,w-1):
                E_H[:,x] = S_H[np.arange(h)+1,x+self.u_arms_L[:,x,d,0]+1] - S_H[np.arange(h)+1,x-self.u_arms_L[:,x,d,1]]
            S_V = np.zeros((h,w))
            for y in range(1,h):
                S_V[y,:] = S_V[y-1,:] + pool[y,:]
            S_V = np.concatenate((np.zeros(w).reshape(1,w),S_V),axis=0)
            S_V = np.concatenate((np.zeros(h+1).reshape(h+1,1),S_V),axis=1)
            E = np.zeros((h,w))
            for y in range(1,h-1):
                E[y,:] = S_V[y+self.u_arms_L[y,:,d,2]+1,np.arange(w)+1] - S_V[y-self.u_arms_L[y,:,d,3],np.arange(w)+1]
            vote[:,:,d] = np.copy(E)
    
            total = np.sum(vote,axis=2)
            elected = np.argmax(vote,axis=2)
            highest_vote = np.max(vote,axis=2)
            vote_rate = np.zeros(total.shape)
            vote_rate[total != 0] = highest_vote[total!=0]/total[total!=0]
            vote_rate[np.isinf(vote_rate)] = 0
    
            new_outlier = np.copy(outlier)
            index = np.logical_and(np.logical_and(total > tao_s, vote_rate > tao_h), outlier == 0)
            new_outlier[index] = 1

            new_D_L = np.copy(D_L)
            new_D_L[index] = elected[index]

        return new_D_L, new_outlier

##################################################

def arm_compute(img):
    '''
    h,w,c = img.shape
    arms = np.ones((h,w,4))
    for y in range(1,h-1):
        for x in range(1,w-1):
            #hp+
            for r in range(2,L1+1):
                if x + r > w - 1:
                    break 
                tao = tao2 if r > L2 else tao1
                if np.max(np.abs(img[y][x] - img[y][x+r])) > tao:
                    break
                arms[y][x][0] = r
            #hp-
            for r in range(2,L1+1):
                if x - r < 0:
                    break 
                tao = tao2 if r > L2 else tao1
                if np.max(np.abs(img[y][x] - img[y][x-r])) > tao:
                    break
                arms[y][x][1] = r
            #vp+
            for r in range(2,L1+1):
                if y + r > h - 1:
                    break 
                tao = tao2 if r > L2 else tao1
                if np.max(np.abs(img[y][x] - img[y+r][x])) > tao:
                    break
                arms[y][x][2] = r
            #vp-
            for r in range(2,L1+1):
                if y - r < 0:
                    break 
                tao = tao2 if r > L2 else tao1
                if np.max(np.abs(img[y][x] - img[y-r][x])) > tao:
                    break
                arms[y][x][3] = r
    '''
    h,w,c = img.shape
    arms = np.ones((h,w,4))
    #hp+
    for x in range(1,w-1):
        mask = np.zeros(h-2,dtype=bool)
        for r in range(2,L1+1):
            if x + r > w - 1:
                break
            tao = tao2 if r > L2 else tao1
            if np.sum(~mask) == 0:
                break
            mask = np.logical_or((np.max(np.abs(img[1:h-1,x] - img[1:h-1,x+r]),axis=1) > tao),mask)
            pad_mask = np.concatenate((np.array([True]),mask,np.array([True])))
            arms[~pad_mask,x,0] = r

    #hp-
    for x in range(1,w-1):
        mask = np.zeros(h-2,dtype=bool)
        for r in range(2,L1+1):
            if x - r < 0:
                break
            tao = tao2 if r > L2 else tao1
            if np.sum(~mask) == 0:
                break
            mask = np.logical_or((np.max(np.abs(img[1:h-1,x] - img[1:h-1,x-r]),axis=1) > tao),mask)
            pad_mask = np.concatenate((np.array([True]),mask,np.array([True])))
            arms[~pad_mask,x,1] = r

    #vp+
    for y in range(1,h-1):
        mask = np.zeros(w-2,dtype=bool)
        for r in range(2,L1+1):
            if y + r > h - 1:
                break
            tao = tao2 if r > L2 else tao1
            if np.sum(~mask) == 0:
                break
            mask = np.logical_or((np.max(np.abs(img[y,1:w-1] - img[y+r,1:w-1]),axis=1) > tao),mask)
            pad_mask = np.concatenate((np.array([True]),mask,np.array([True])))
            arms[y,~pad_mask,2] = r

    #vp-
    for y in range(1,h-1):
        mask = np.zeros(w-2,dtype=bool)
        for r in range(2,L1+1):
            if y - r < 0:
                break
            tao = tao2 if r > L2 else tao1
            if np.sum(~mask) == 0:
                break
            mask = np.logical_or((np.max(np.abs(img[y,1:w-1] - img[y-r,1:w-1]),axis=1) > tao),mask)
            pad_mask = np.concatenate((np.array([True]),mask,np.array([True])))
            arms[y,~pad_mask,3] = r
            
    return arms

def penalty(v1_L,v2_L,v1_R,v2_R):
    dim,c = v1_L.shape
    P1 = np.zeros(dim)
    P2 = np.zeros(dim)
    D1 = np.max(np.abs(v1_L-v2_L),axis=1)
    D2 = np.max(np.abs(v1_R-v2_R),axis=1)
    P1[np.logical_and(D1<tao_so,D2<tao_so)] = G1
    P2[np.logical_and(D1<tao_so,D2<tao_so)] = G2
    P1[np.logical_and(D1<tao_so,D2>tao_so)] = G1/4
    P2[np.logical_and(D1<tao_so,D2>tao_so)] = G2/4
    P1[np.logical_and(D1>tao_so,D2<tao_so)] = G1/4
    P2[np.logical_and(D1>tao_so,D2<tao_so)] = G2/4
    P1[np.logical_and(D1>tao_so,D2>tao_so)] = G1/10
    P2[np.logical_and(D1>tao_so,D2>tao_so)] = G2/10
    return P1,P2

def penalty_2(v1_L,v2_L,v1_R,v2_R):
    dim,c = v1_L.shape
    P1 = np.zeros(dim)
    P2 = np.zeros(dim)
    D1 = np.max(np.abs(v1_L-v2_L),axis=1)
    D2 = np.max(np.abs(v1_R-v2_R),axis=1)
    P1[np.logical_and(D1<tao_so,D2<tao_so)] = G1
    P2[np.logical_and(D1<tao_so,D2<tao_so)] = G2
    P1[np.logical_and(D1<tao_so,D2>tao_so)] = G1/4
    P2[np.logical_and(D1<tao_so,D2>tao_so)] = G2/4
    P1[np.logical_and(D1>tao_so,D2<tao_so)] = G1/4
    P2[np.logical_and(D1>tao_so,D2<tao_so)] = G2/4
    P1[np.logical_and(D1>tao_so,D2>tao_so)] = G1/10
    P2[np.logical_and(D1>tao_so,D2>tao_so)] = G2/10
    return P1,P2

def subpixel_enhancement(label, cost):
    result = np.empty(label.shape)
    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            disp = label[y, x].astype(np.int32)
            result[y, x] = disp
            if 1 <= disp and disp < cost.shape[2] - 1:
                cn = cost[y, x, disp - 1]
                cz = cost[y, x, disp]
                cp = cost[y, x, disp + 1]
                denominator = 2 * (cp + cn - 2 * cz)
                if denominator > 1e-5:
                    result[y, x] = disp - min(1, max(-1, (cp - cn) / denominator))
    return result