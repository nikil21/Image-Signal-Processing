# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:57:30 2021

@author: ideapad
"""
#Import libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np

#Load image
src_img = cv2.imread("Globe.png",0)
N = src_img.shape[0]

def varblur(src_img) :
    Nr,Nc = src_img.shape
    fin = np.zeros(src_img.shape)

    #Go along rows and then along columns
    kmax = 13
    for i in range(kmax,Nr+kmax) :
        for j in range(kmax,Nc+kmax) :
            #Obtain blur kernel-sigma values
            A = 2
            B = (N*N/2)/(-np.log(0.01/A))  
            ioff = i-kmax
            joff = j-kmax
            sig = A*np.exp(-((ioff-(N/2))**2+(joff-(N/2))**2)/B)
            #sig = 1
            
            #Apply kernel on the image
            kernel = generate_kernel(sig)
            kext = len(kernel)//2
            
            img = np.zeros((int(Nr+2*kmax),int(Nc+2*kmax)))
            
            img[kmax:Nr+kmax,kmax:Nc+kmax] = src_img
            patch = np.zeros(kernel.shape)
            patch = img[i-kext:i+kext+1,j-kext:j+kext+1]
            patch = patch*kernel
            fin[i-kmax,j-kmax] = sum(sum(patch))
  
    return fin

#Obtain kernel for given sigma    
def generate_kernel(sig) :
    k = int(np.ceil(6*sig +1))
    if k%2 == 0 :
        k = k+1
    kernel = np.zeros((k,k))
    mid = k//2
    for i in range(0,mid+1) : 
        row = np.arange(mid+i,k)
        roweff = row-mid
        kernel[mid-i,row] = (1/(2*np.pi*sig*sig))*np.exp(-(roweff*roweff + i*i)/(2*sig*sig))
        kernel[mid-roweff[1:],mid+i] = kernel[mid-i,row][1:]
    
    kernel[:mid+1,:mid] = np.fliplr(kernel[:mid+1,mid+1:])  
    kernel[mid+1:,:] = np.flipud(kernel[:mid,:])
    kernel = kernel/sum(sum(kernel))
    return kernel 
    
#Obtain target
tgt = varblur(src_img)
#Plots
plt.imshow(tgt, cmap = "gray")
plt.axis("off")
plt.show()

