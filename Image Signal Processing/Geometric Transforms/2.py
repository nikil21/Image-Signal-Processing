#Import libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np



src = cv2.imread("pisa_rotate.png")
print(src)
print(src.shape)

#Initialize tgt
tgt = np.zeros((src.shape))
print(len(tgt),len(tgt[0]))

#Bilinear function
def bilnr(src,theta, xn, yn) :
     #Mapping to src grid
	x = np.cos(theta)*xn - np.sin(theta)*yn
	y = np.sin(theta)*xn + np.cos(theta)*yn
	xf = int(np.floor(x)) + x_cen
	yf = int(np.floor(y)) + y_cen
	
	#distance from pixel
	a = x+ x_cen - xf
	b = y+ y_cen - yf
	
	#print(xf)
	
	#Calculate intensity
	if xf >= (src.shape)[0]-1 or yf >= (src.shape)[1]-1 or xf<=0 or yf<= 0 :
	     Ival = 0
	else :
	     Ival = (1-a)*(1-b)*src[xf][yf] + (1-a)*(b)*src[xf][yf+1] + (a)*(1-b)*src[xf+1][yf] + (a)*(b)*src[xf+1][yf+1]
	     #Ival = src[xf][yf]
	return Ival



#Define centre
x_cen = int(np.floor((src.shape)[0]/2))
y_cen = int(np.floor((src.shape)[1]/2))
#Translation
theta = -4* np.pi/180



for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :

          tgt[xn][yn] = bilnr(src, theta, xn-x_cen, yn-y_cen)
          

#FINAL OUTPUT HAS A ROTATION OF 4 DEGREES ABOUT THE CENTRE COMPARED TO THE ORIGINAL IMAGE
plt.imshow(tgt,cmap = "gray")
plt.show()
print(tgt)
print(tgt.shape)
