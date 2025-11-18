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
                if (ztemp < zbufer[y, x]):
                    image[y, x] = color
                    zbufer[y, x] = ztemp
               
def rotation_matrix (alpha, beta, gamma): 
    Rx = np.array([[1, 0 , 0],
                  [0, math.cos(alpha), math.sin(alpha)],
                  [0, -math.sin(alpha), math.cos(alpha)]
                  ]) 
    Ry = np.array([[math.cos(beta), 0 , math.sin(beta)],
                  [0, 1, 0],
                  [-math.sin(beta), 0, math.cos(beta)]]) 
    Rz = np.array([[math.cos(gamma), math.sin(gamma), 0],
                  [-math.sin(gamma), math.cos(gamma), 0],
                  [0, 0 , 1]]) 
    return Rx @ Ry @ Rz
                  
    
def conversion_screen_coordinates_of_form(vex, ax, ay, v0, u0):
    X, Y, Z = vex
    u = (ax * X) / Z + u0
    v = (ay * Y) / Z + v0    
    return u, v
    
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
        
R = rotation_matrix(0, 180, 0)
translation = np.array([0.005, -0.035, 0.15])

for m in range(len(v)):
    v[m] @= R
    v[m] += translation
    
for k in range(len(f)):
    temp0 = v[f[k][0] - 1]
    temp1 = v[f[k][1] - 1]
    temp2 = v[f[k][2] - 1]
    
    x0 = temp0[0]
    x1 = temp1[0]
    x2 = temp2[0]
    y0 = temp0[1]
    y1 = temp1[1]
    y2 = temp2[1]
    z0 = temp0[2]
    z1 = temp1[2]
    z2 = temp2[2]
    
    n = np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 -y0, z1 - z0])
    cos_osvechenia = (np.dot(n, [0, 0, 1])) / (np.linalg.norm(n))
    
    conversionToScreen0 = conversion_screen_coordinates_of_form(temp0, 2000, 2000, 1000, 1000)
    conversionToScreen1 = conversion_screen_coordinates_of_form(temp1, 2000, 2000, 1000, 1000)
    conversionToScreen2 = conversion_screen_coordinates_of_form(temp2, 2000, 2000, 1000, 1000)
    
    x0 = conversionToScreen0[0]
    y0 = conversionToScreen0[1]
        
    x1 = conversionToScreen1[0]
    y1 = conversionToScreen1[1]
     
    x2 = conversionToScreen2[0]
    y2 = conversionToScreen2[1]
    
    
    
    if (cos_osvechenia < 0):
        drawing_triangles(img_mat, (-255*cos_osvechenia, 0, 0), x0, y0, x1, y1, x2, y2, z0, z1, z2)
    

img = ImageOps.flip(Image.fromarray(img_mat, mode = 'RGB'))
img.save("img.png")