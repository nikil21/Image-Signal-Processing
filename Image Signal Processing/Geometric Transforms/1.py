# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 19:22:40 2021

@author: ideapad
"""
#Import libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np

#Bilinear function
def bilnr(src,tx, ty, xn, yn) :
     #Mapping to src grid
	x = xn-(tx)
	y = yn-ty
	xf = int(np.floor(x))
	yf = int(np.floor(y))
	
	#distance from pixel
	a = x-xf
	b = y-yf
	
	#Calculate intensity
	if xf >= (src.shape)[0]-1 or yf >= (src.shape)[1]-1 or xf<=0 or yf<= 0 :
	     Ival = 0
	else :
	     Ival = (1-a)*(1-b)*src[xf][yf] + (1-a)*(b)*src[xf][yf+1] + (a)*(1-b)*src[xf+1][yf] + (a)*(b)*src[xf+1][yf+1]
	
	return Ival

#Define path of src
#path  = r"lena_translate.pgm"
src = cv2.imread("lena_translate.png")
print(src.shape)

cv2.imshow("Source",src)
cv2.waitKey(0)

#Translation
tx = 3.75
ty = 4.3

#Initialize tgt
tgt = np.zeros(src.shape)
print(tgt)
print(len(tgt),len(tgt[0]))

for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :
          tgt[xn][yn] = bilnr(src, tx, ty, xn, yn)
          
#print(tgt[:6])
#cv2.imwrite("image",tgt)
#cv2.imread("image
plt.imshow(tgt,cmap = "gray")
plt.show()
print(tgt)