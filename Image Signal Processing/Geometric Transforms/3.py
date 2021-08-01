#Import libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np

src = cv2.imread("cells_scale.png")
cv2.imshow('source', src)

#Define centre
x_cen = int(np.floor((src.shape)[0]/2))
y_cen = int(np.floor((src.shape)[1]/2))

#Translation
x_scl = 1.2
y_scl = 1.2

#Initialize tgt
tgt = np.zeros((src.shape))


#Bilinear function
def bilnr(src, x_scl, y_scl, xn, yn) :
     #Mapping to src grid
	x = xn/x_scl
	y = yn/y_scl
	xf = int(np.floor(x)) + x_cen
	yf = int(np.floor(y)) + y_cen
	
	#distance from pixel
	a = x+ x_cen - xf
	b = y+ y_cen - yf
    
	#Calculate intensity
	if xf >= (src.shape)[0]-1 or yf >= (src.shape)[1]-1 or xf<=0 or yf<= 0 :
	     Ival = 0
	else :
	     Ival = (1-a)*(1-b)*src[xf][yf] + (1-a)*(b)*src[xf][yf+1] + (a)*(1-b)*src[xf+1][yf] + (a)*(b)*src[xf+1][yf+1]
	return Ival



for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :
          tgt[xn][yn] = bilnr(src, x_scl, y_scl, xn-x_cen, yn-y_cen)
          
#print(tgt)
plt.imshow(tgt,cmap = "gray")
plt.show()
print(tgt)
