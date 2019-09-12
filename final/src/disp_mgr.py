import numpy as np
import cv2
from disparity import disp_calculator as dc
from disparity import subpixel_enhancement as sub
from jerry import *

class dispMgr:

	def __init__(self,imgL,imgR):
		self.imgL = imgL
		self.imgR = imgR
		self.h, self.w, self.ch = imgL.shape
		self.posDC = dc(imgL,imgR)
		self.negDC = dc(imgR,imgL)
		self.isReal, min_disp, max_disp = test_range(imgR,imgL)
		print((-max_disp,-min_disp))
		self.pos_disp = 47 if -min_disp < 47 and not self.isReal else -min_disp
		#print(self.pos_disp)
		self.pos_disp += 5
		self.neg_disp = max_disp + 5
		self.disparityL = np.zeros((imgL.shape[0],imgL.shape[1]))
		self.disparityR = np.zeros((imgL.shape[0],imgL.shape[1]))
		self.posDC.set_max_disparity(self.pos_disp)
		self.negDC.set_max_disparity(self.neg_disp)

	def computeDisp(self):
		self.posDC.set_arms()
		self.posDC.build_census_window(11,11)
		self.pos_cost_L = self.posDC.find_L_disparity()
		self.pos_cost_R = self.posDC.find_R_disparity()
		cost_L = np.copy(self.pos_cost_L)
		cost_R = np.copy(self.pos_cost_R)
		if self.isReal:
			self.negDC.set_arms()
			self.negDC.build_census_window(11,11)
			self.neg_cost_L = self.negDC.find_R_disparity()
			self.neg_cost_R = self.negDC.find_L_disparity()

			cost_L = self.cost_concatenate_L()
			cost_R = self.cost_concatenate_R()

		self.disparityL = self.find_optimal_disparity(cost_L)
		self.disparityR = self.find_optimal_disparity(cost_R)

		self.disparityL = self.border_refinement(self.disparityL)
		self.disparityR = self.border_refinement(self.disparityR)
		self.disparityL = cv2.medianBlur(self.disparityL.astype('uint8'),3).astype('int')
		self.disparityR = cv2.medianBlur(self.disparityR.astype('uint8'),3).astype('int')

		if self.isReal:
			self.disparityL -= (self.neg_disp - 1)
			self.disparityR -= (self.neg_disp - 1)
		
		self.find_outlier()
		#if self.isReal:
		#	self.disparityL += (self.neg_disp - 1)
		if not self.isReal:
			d_max = self.pos_disp
			self.disparityL, self.outlier = self.posDC.region_vote(self.disparityL,self.outlier,d_max)
			self.disparityL, self.outlier = self.posDC.region_vote(self.disparityL,self.outlier,d_max)
			self.disparityL, self.outlier = self.posDC.region_vote(self.disparityL,self.outlier,d_max)
			self.disparityL, self.outlier = self.posDC.region_vote(self.disparityL,self.outlier,d_max)
			self.disparityL, self.outlier = self.posDC.region_vote(self.disparityL,self.outlier,d_max)
			self.find_outlier()
			self.fill_outlier()

		lower_bound = -self.neg_disp if self.isReal else 5
		self.disparityL = np.clip(self.disparityL,lower_bound,self.pos_disp)

		if self.isReal:
			self.disparityL += (self.neg_disp - 1)

		self.segmentation(200,200,True)
		
		if self.isReal:
			self.disparityL -= (self.neg_disp - 1)
		
		self.disparityL = np.clip(self.disparityL,lower_bound,self.pos_disp)
		'''
		L = np.copy(self.disparityL)
		L += (self.neg_disp - 1)
		print(np.bincount(L.flatten().astype('int')))
		L = L - np.min(L)
		L = (L / np.max(L)).astype('float32')
		L *= 255
		L = cv2.applyColorMap(L.astype('uint8'), cv2.COLORMAP_JET)
		cv2.imshow('L',L)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		'''
		self.find_outlier()


		if not self.isReal:
			self.disparityL = sub(self.disparityL, cost_L)

		return self.disparityL.astype('float32'), (self.outlier*255).astype('uint8')

	def find_optimal_disparity(self,costs):
		return np.argmin(costs,axis=2)

	def cost_concatenate_L(self):
		neg_cost = np.flip(self.neg_cost_L,axis=2)[:,:,:-1]
		return np.concatenate((neg_cost,self.pos_cost_L),axis=2)

	def cost_concatenate_R(self):
		neg_cost = np.flip(self.neg_cost_R,axis=2)[:,:,:-1]
		return np.concatenate((neg_cost,self.pos_cost_R),axis=2)

	def find_outlier(self,dilate=True):
		D_L = np.copy(self.disparityL).astype('int')
		D_R = np.copy(self.disparityR).astype('int')
		#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
		outlier = np.zeros((self.h,self.w))
		for x in range(self.w):
			for y in range(self.h):
				if(x-D_L[y,x] >= 0 and x-D_L[y,x] < self.w):
					if(np.abs(D_L[y,x]-D_R[y,x-D_L[y,x]]) < 1.5):
						outlier[y,x] = 1
						#outlier[np.logical_and(np.logical_and(np.abs(D_L[:,x] - D_R[np.arange(h),x-D_L[:,x]]) < 1.5, x-D_L[:,x] >= 0),np.sum(np.abs(imgR[np.arange(h),x-D_L[:,x]] - imgL[:,x]),axis=1) < 50),x] = 1
		self.outlier = outlier

	def fill_outlier(self):
		new_disparity = np.copy(self.disparityL)
		for x in range(100,-1,-1):
			new_disparity[self.outlier[:,x] == 0,x] = new_disparity[self.outlier[:,x] == 0,x+1]
		self.disparityL = new_disparity

	def border_refinement(self,disp):
		img_valid = disp[2:self.h-2,5:self.w-2].astype('uint8')
		new_img = cv2.copyMakeBorder(img_valid,2,2,5,2,cv2.BORDER_REPLICATE)
		return new_img.astype('int')

	def segmentation(self,k,min_size,bi):
		# Graph-Based Image Segmentation 分割器
		segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=k, min_size=min_size)	

		# 分割圖形
		segment = segmentator.processImage(self.imgL)	

		d = np.copy(self.disparityL).astype('int')

		for i in range(np.max(segment)):
			valid = np.logical_and(self.outlier == 1, segment == i)
			invalid = np.logical_and(self.outlier == 0, segment == i)

			if(d[valid].size != 0 and d[invalid].size != 0 and d[valid].size/(d[valid].size+d[invalid].size) > 0.15):
				vote = np.bincount(d[valid])
				index = np.argwhere(vote == np.max(vote))
				mean = np.mean(d[valid])
				new_d = 255
				for j in index:
					if j - mean < new_d:
						new_d = j
				d[invalid] = new_d

			if(d[invalid].size/(d[valid].size+d[invalid].size) > 0.95 and np.max(np.where(np.logical_or(valid,invalid) == True)[1]) < np.max(self.disparityL) + 5):
				d[segment == i] = int((self.pos_disp) * 0.9)

		if bi:
			d = cv2.bilateralFilter(d.astype('uint8'),10,9,2).astype('float32')
		
		self.disparityL = d
