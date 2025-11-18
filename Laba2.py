from math import cos
from math import sin
import random
import math
import numpy as np

from PIL import Image, ImageOps
img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8) 
zbufer = np.zeros((2000, 2000), dtype = np.float64)  
 

def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):      
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    
    return lambda0, lambda1, lambda2
    
def drawing_triangles(image, color, x0, y0, x1, y1, x2, y2, z0, z1, z2):
    xmin = max(0, int(min(x0, x1, x2)))
    xmax = min(2000, int(max(x0, x1, x2)) + 1)
    ymin = max(0, int(min(y0, y1, y2)))  
    ymax = min(2000, int(max(y0, y1, y2)) + 1)   
    
    for x in range (xmin, xmax):
        for y in range (ymin, ymax):
            lambda0, lambda1, lambda2 = barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
            if(lambda0 >= 0 and lambda1>= 0  and lambda2 >= 0): 
                ztemp = lambda0*z0 + lambda1*z1 + lambda2*z2
                if (ztemp < zbufer[x][y]):
                    image[x, y] = color
                    zbufer[x][y] = ztemp
    
def draw_line7(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
        
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0    
        
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    
    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color 
        
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2.0*(x1 - x0)
            y += y_update    
        
for i in range(2000):
    for j in range(2000):
        img_mat [i, j] = 0, 255, 127
        
for i in range(2000):
    for j in range(2000):
        zbufer[i][j] = np.inf


file=open('model_1.obj')
v=[]
f=[]
for s in file:
    sp=s.split()
    if(sp[0]=='v'):
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])
    if(sp[0] == 'f'):
        f.append([int(sp[1].split('/')[0]), int(sp[2].split('/')[0]), int(sp[3].split('/')[0])])

for k in range(len(f)): 
    x0 = 8000 * v[f[k][0] - 1][0] + 1000
    y0 = 8000 * v[f[k][0] - 1][1] + 1000
    z0 = 8000 * v[f[k][0] - 1][2] + 1000
    
    x1 = 8000 * v[f[k][1] - 1][0] + 1000
    y1 = 8000 * v[f[k][1] - 1][1] + 1000
    z1 = 8000 * v[f[k][1] - 1][2] + 1000
    
    x2 = 8000 * v[f[k][2] - 1][0] + 1000
    y2 = 8000 * v[f[k][2] - 1][1] + 1000
    z2 = 8000 * v[f[k][2] - 1][2] + 1000
    
    n = np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 -y0, z1 - z0])
    cos_osvechenia = (np.dot(n, [0, 0, 1])) / (np.linalg.norm(n))
    
    if (cos_osvechenia < 0):
        drawing_triangles(img_mat, (-255*cos_osvechenia, 0, 0), x0, y0, x1, y1, x2, y2, z0, z1, z2)
    

img = Image.fromarray(img_mat, mode = 'RGB')
img.save("img.png")