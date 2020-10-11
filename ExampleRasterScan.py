# Single pixel imaging in python

from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image, ImageOps
from scipy.linalg import hadamard

# Read in and resize image
sz = 16 #Size of image to use (must be n^2 for Hadamard)
path = "Teddy256.png"
raw_img = cv2.imread(path)
gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.resize(gray_img, (sz,sz)) 

# Convert to number array
obj = np.array(gray_img)

# Make measurements
##obj_Vector = np.reshape(obj, (1,sz**2)) #Reshape to a vector to make maths easier
I = np.identity(sz**2)
H = hadamard(sz**2)
m = np.zeros((sz**2,1))

PatternSet  =  I; # Chose H or I here.

# Measure for each pattern
for i in range(0,sz**2):
    samplingVector = PatternSet[i,:] 
    samplingPattern = np.reshape(samplingVector,(sz,sz))
    plt.imshow(samplingPattern)
    plt.show()
    #time.sleep(0.005)
    m[i] = np.sum(samplingPattern * obj)

# Reconstruct Image
recon = np.matmul(PatternSet,m)
recon = np.reshape(recon, (sz,sz))
fig = plt.imshow(recon) 

