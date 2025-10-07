from math import cos
from math import sin
import math
import numpy as np
from PIL import Image, ImageOps
img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

#Первая реализация звездочки
def draw_line0(image, x0, y0, x1, y1, color):
    step = 1.0/100
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color
        
#Вторая реализация звездочки
def draw_line1(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t) * x0 + t * x1)
        y = round ((1.0 - t) * y0 + t * y1)
        image[y, x] = color
     
#Реализация половины зведзды        
def draw_line2(image, x0, y0, x1, y1, color):
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color
        
#Реализация звездочек побокам
def draw_line3(image, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color
        
#Реализация 1/4 звездочки      
def draw_line4(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color      
            
#Нормисная реализация звездочки
def draw_line5(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
        
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0    
        
    y = y0
    dy = abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    
    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color 
        
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update
 
#Нормисная реализация звездочки с умножением на 2 (х0 - х1)         
def draw_line6(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
        
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0    
        
    y = y0
    dy = 2.0*(x1 - x0)*abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    
    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color 
        
        derror += dy
        if (derror > 2.0*(x1 - x0)*0.5):
            derror -= 2.0*(x1 - x0)*1.0
            y += y_update            
    
#Нормисная реализация звездочки с сокращениями
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
"""
for k in range(200):
    x0, y0 = 100, 100
    x1 = 100 + 95 * cos((2 * 3.14) / 13 * k)
    y1 = 100 + 95 * sin((2 * 3.14) / 13 * k)
    draw_line0(img_mat, x0, y0, x1, y1, (255, 255, 0))
        
for k in range(200):
    x0, y0 = 250, 50
    x1 = 250 + 25 * cos((2 * 3.14) / 13 * k)
    y1 = 50 + 25 * sin((2 * 3.14) / 13 * k)
    draw_line1(img_mat, x0, y0, x1, y1, (255, 255, 0))
        
for k in range(200):
    x0, y0 = 300, 100
    x1 = int (300 + 25 * cos((2 * 3.14) / 13 * k))
    y1 = int (100 + 25 * sin((2 * 3.14) / 13 * k))
    draw_line5(img_mat, x0, y0, x1, y1, (255, 255, 0))
      
for k in range(200):
    x0, y0 = 300, 170
    x1 = int (300 + 25 * cos((2 * 3.14) / 13 * k))
    y1 = int (170 + 25 * sin((2 * 3.14) / 13 * k))
    draw_line6(img_mat, x0, y0, x1, y1, (255, 255, 0))
    
for k in range(200):
    x0, y0 = 250, 220
    x1 = int (250 + 25 * cos((2 * 3.14) / 13 * k))
    y1 = int (220 + 25 * sin((2 * 3.14) / 13 * k))
    draw_line7(img_mat, x0, y0, x1, y1, (255, 255, 0))
"""            

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
    draw_line7(img_mat, int(8000*v[f[k][0] - 1][0]+ 1000), int(8000*v[f[k][0] - 1][1]+ 1000), int(8000*v[f[k][1] - 1][0]+ 1000), int(8000*v[f[k][1] - 1][1]+ 1000), (75, 0, 130)) 
    draw_line7(img_mat, int(8000*v[f[k][0] - 1][0]+ 1000), int(8000*v[f[k][0] - 1][1]+ 1000), int(8000*v[f[k][2] - 1][0]+ 1000), int(8000*v[f[k][2] - 1][1]+ 1000), (106, 90, 205)) 
    draw_line7(img_mat, int(8000*v[f[k][1] - 1][0]+ 1000), int(8000*v[f[k][1] - 1][1]+ 1000), int(8000*v[f[k][2] - 1][0]+ 1000), int(8000*v[f[k][2] - 1][1]+ 1000), (25, 25, 112)) 


img = Image.fromarray(img_mat, mode = 'RGB')
img.save("img.png")