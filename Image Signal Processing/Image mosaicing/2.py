# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 22:19:31 2021

@author: ideapad
"""
#Import libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random as rn
import csv
from scipy.linalg import null_space
#Functions 

#Bilinear
#Rows and columns of the target as input
def bilnr(src, H, rows, cols) :
     #Creating vector to multiply Hinv
     x = list(np.arange(0,rows))
     x = x*cols
     x = np.array(x)
     x = x.reshape(cols,rows)
     x = x.T
     x = x.reshape(int(rows*cols),1)
     y = list(np.arange(0,cols))
     y = y*rows
     y = np.array(y)
     y = y.reshape(int(rows*cols),1)
     o = np.ones((int(rows*cols),1))
     xy = np.array([x,y,o])
     xy = xy.T
     xy = xy[0]
     xy = xy.T
     xy[1] = xy[1] - cenx
     
     #Target to source mapping  
     xy_temp = np.linalg.inv(H)@ xy
     #xy_temp = xy_temp.T
     x = xy_temp[0]/xy_temp[2]
     y = xy_temp[1]/xy_temp[2]
     
     #print(shape(x),shape(y))
     
     #xf = int(np.floor(x)) 
     #yf = int(np.floor(y))
     xf = x.astype(int)
     yf = y.astype(int)
	
	#distance from pixel
     a = x-xf
     b = y-yf
     
     '''
     src_vec = src.reshape(int(len(src)*len(src[0])),1)
     geqx = (shape(src)[0]-1)*np.ones(shape(xf))
     leqx = np.zeros(shape(xf))
     geqy = (shape(src)[1]-1)*np.ones(shape(yf))
     leqy = np.zeros(shape(yf))

     val_geqx = np.greater_equal(xf,geqx)
     val_leqx = np.less_equal(xf,leqx)
     val_geqy = np.greater_equal(yf,geqy)
     val_leqy = np.less_equal(yf,leqy)
     check = val_geqx + val_geqy + val_leqx + val_leqy
     '''
     
     Ival = np.zeros(xf.shape)
     #print(shape(src))
     #Find intensity
     for i in range(0,len(xf)) :
          #if check[i] == False :
          if xf[i] < src.shape[0]-1 and yf[i] < src.shape[1]-1 and xf[i]>0 and yf[i]>0 :
               #print(yf[i])
               Ival[i] = (1-a[i])*(1-b[i])*src[xf[i]][yf[i]] + (1-a[i])*(b[i])*src[xf[i]][yf[i]+1] + (a[i])*(1-b[i])*src[xf[i]+1][yf[i]] + (a[i])*(b[i])*src[xf[i]+1][yf[i]+1]

     Ival = Ival.reshape(rows,cols)
     return Ival

#RANSAC
def ransac(corresp1,corresp2) :
     frac = 0
     niter = 0
     while(frac <= 0.75) :
             
          #Generate 4 random numbers from the set
          Lmp = len(corresp1)
          r = rn.sample(range(0,Lmp),4)
          a = [corresp1[r[i]] for i in range(0,len(r))]
          b = [corresp2[r[i]] for i in range(0,len(r))] 
          #Take these 4 points and find homography
          #Fill in the matrix
          vsize = 9
          eqns = 4
          A = np.zeros((int(2*eqns),vsize))
          #print(shape(A))
          #Loop to fill in the values
          for i in range(0,eqns) :
               A[int(2*i)][0] = b[i][0]
               A[int(2*i)][1] = b[i][1]
               A[int(2*i)][2] = 1
               A[int(2*i)][3] = 0
               A[int(2*i)][4] = 0
               A[int(2*i)][5] = 0
               A[int(2*i)][6] = -b[i][0]*a[i][0]
               A[int(2*i)][7] = -b[i][1]*a[i][0]
               A[int(2*i)][8] = -a[i][0]
               
               A[int(2*i)+1][0] = 0
               A[int(2*i)+1][1] = 0
               A[int(2*i)+1][2] = 0
               A[int(2*i)+1][3] = b[i][0]
               A[int(2*i)+1][4] = b[i][1]
               A[int(2*i)+1][5] = 1
               A[int(2*i)+1][6] = -b[i][0]*a[i][1]
               A[int(2*i)+1][7] = -b[i][1]*a[i][1]
               A[int(2*i)+1][8] = -a[i][1]

          #Find nullspace of the matrix
          h = null_space(A)
          #print(shape(h))
          #Put h in order
          H = h.reshape((3,3))

          #Check with remaining points and see fraction
          C = []
          iterset = list(set(np.arange(0,Lmp)).difference(r))
          bvec = np.zeros((3,1))
          avec = np.zeros((2,1))
          eps = 12
          #12 0.75 good
          #16 0.75 works
          bvec[2] = 1
          for item in iterset :
               bvec[0] = corresp2[item][0]
               bvec[1] = corresp2[item][1]
               
               atemp = H @ bvec
               avec[0] = atemp[0]/atemp[2]
               avec[1] = atemp[1]/atemp[2]
               
               dist = np.sqrt(pow(corresp1[item][0]-avec[0],2) + pow(corresp1[item][1]-avec[1],2))
               if dist < eps :
                    C.append(item)
               
          #Check how good in the consensus set
          frac = len(C)/len(iterset)
          niter = niter+1
     return H,frac,niter,C      

#Define path of src

#path1  = r"/home/vignesh/EE5175/Lab2/room1.jpeg"
src1 = cv2.imread("wall1.png",0)
print(src1.shape)

#path2  = r"/home/vignesh/EE5175/Lab2/room2.jpeg"
src2 = cv2.imread("wall2.png",0)

#path3  = r"/home/vignesh/EE5175/Lab2/room3.jpeg"
src3 = cv2.imread("wall3.png",0)

#Run SIFT and obtain matching key points
correspa1 = []
correspa2 = []
with open("vig_mosaic2_1.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        correspa1.append(row[:2])
        correspa2.append(row[2:])
correspa1 = np.array(correspa1)
correspa2 = np.array(correspa2)

#Run SIFT and obtain matching key points
correspc1 = []
correspc2 = []
with open("vig_mosaic2_3.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        correspc1.append(row[:2])
        correspc2.append(row[2:])
correspc1 = np.array(correspc1)
correspc2 = np.array(correspc2)

H1,frac1,niter1,C1 = ransac(correspa1,correspa2)
H3,frac3,niter3,C3 = ransac(correspc1,correspc2)
print(frac1,frac3)
#print(H1,H3)
#print(niter1,niter3)

#Define topcorer
#print(shape(src1))
cenx = int(np.floor(src1.shape[1]))

#Create canvas
nrows = src2.shape[0]
ncolumns = src1.shape[1] + src2.shape[1] + src3.shape[1]
print(src1.shape[1],src2.shape[1])

canvas = np.zeros((nrows,ncolumns))
countcnvs = np.zeros((nrows,ncolumns))

canvas1 = bilnr(src1, H1, nrows, ncolumns)
canvas2 = bilnr(src2, np.identity(3), nrows, ncolumns)
canvas3 = bilnr(src3, H3, nrows, ncolumns)

#Finding no of intensities at each point
temp = np.equal(canvas1,np.zeros(canvas1.shape))
temp = ~temp
temp = temp.astype(int)
countcnvs = countcnvs + temp

temp = np.equal(canvas2,np.zeros(canvas1.shape))
temp = ~temp
temp = temp.astype(int)
countcnvs = countcnvs + temp

temp = np.equal(canvas3,np.zeros(canvas1.shape))
temp = ~temp
temp = temp.astype(int)
countcnvs = countcnvs + temp

#Image plot
canvas = canvas1 + canvas3 + canvas2 
plt.imshow(canvas/countcnvs,cmap = "gray")
plt.axis("off")
plt.show()

