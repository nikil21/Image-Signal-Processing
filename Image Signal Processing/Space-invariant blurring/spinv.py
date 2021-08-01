# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:53:19 2021

@author: ideapad
"""
#Import libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np

#Load image
src_img = cv2.imread("Mandrill.png",0)
sigma = np.array([1.6, 1.2, 1, 0.6, 0.3, 0])



def invblur(src_img,kernel) :
    
    kext = len(kernel)//2
    Nr,Nc = src_img.shape
    img = np.zeros((int(Nr+2*kext),int(Nc+2*kext)))
    fin = np.zeros(src_img.shape)
    img[kext:Nr+kext,kext:Nc+kext] = src_img
    patch = np.zeros(kernel.shape)

    #Go along rows and then along columns
    for i in range(kext,Nr+kext) :
        for j in range(kext,Nc+kext) :
            patch = img[i-kext:i+kext+1,j-kext:j+kext+1]
            patch = patch*kernel
            #print(sum(patch))
            fin[i-kext,j-kext] = sum(sum(patch))
            
    return fin

#Obtain kernel for given sigma    
def generate_kernel(sig) :
    k = int(np.ceil(6*sig +1))
    kernel = np.zeros((k,k))
    mid = k//2
    for i in range(0,mid+1) : 
        row = np.arange(mid+i,k)
        roweff = row-mid
        kernel[mid-i,row] = (1/(2*np.pi*sig*sig))*np.exp(-(roweff*roweff + i*i)/(2*sig*sig))
        kernel[mid-roweff[1:],mid+i] = kernel[mid-i,row][1:]
    
    kernel[:mid+1,:mid] = np.fliplr(kernel[:mid+1,mid+1:])  
    kernel[mid+1:,:] = np.flipud(kernel[:mid,:])
    kernel = kernel/sum(kernel)
    return kernel 
    

for i in range(0,len(sigma)) :
    sig = sigma[i]
    if sig == 0 :
        plt.imshow(src_img, cmap = "gray")
        plt.show()
    else :  
        kernel = generate_kernel(sig)
        tgt = invblur(src_img,kernel)
        #Plots
        plt.imshow(tgt, cmap = "gray")
        plt.axis("off")
        plt.show()
        
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
    kernel = kernel/sum(kernel)
    return kernel 
