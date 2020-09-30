# Single pixel imaging in python

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.linalg import hadamard

sz = 16 #Size of image to use (must be n^2 for Hadamard)
path = "Teddy256.png"

# Open image, resize and make gray scale
raw_img = Image.open(path)
gray_img = ImageOps.grayscale(raw_img)
gray_img = gray_img.resize((sz,sz)) 

# Convert to number array
obj = np.array(gray_img)
#plt.imshow(obj)

# Make measurements
obj_Vector = np.reshape(obj, (1,sz**2)) #Reshape to a vector to make maths easier
H = hadamard(sz**2)
m = np.zeros((sz**2,1))

# Measure for each pattern
for i in range(0,sz**2):
    m[i] = np.sum(H[i,:] * obj_Vector)

# Reconstruct Image
recon = np.matmul(H,m)
recon = np.reshape(recon, (sz,sz))
fig = plt.imshow(recon) 