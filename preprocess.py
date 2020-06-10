'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''

# 本质：统计每个像素灰度 出现的概率 0-255 p
# 累计概率
# 1 0.2  0.2
# 2 0.3  0.5
# 3 0.1  0.6
# 256
# 100 0.5 255*0.5 = new
# 1 统计每个颜色出现的概率 2 累计概率 1 3 0-255 255*p
# 4 pixel
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('E:/my research/fish identification/SESSION1_ROI/003/fish_2_003_10.png', 1)
cv2.imshow('src', img)

imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]

count_b = np.zeros(256, np.float)
count_g = np.zeros(256, np.float)
count_r = np.zeros(256, np.float)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        index_b = int(b)
        index_g = int(g)
        index_r = int(r)
        count_b[index_b] = count_b[index_b] + 1
        count_g[index_g] = count_g[index_g] + 1
        count_r[index_r] = count_r[index_r] + 1
for i in range(0, 255):
    count_b[i] = count_b[i] / (height * width)
    count_g[i] = count_g[i] / (height * width)
    count_r[i] = count_r[i] / (height * width)
# 计算累计概率
sum_b = float(0)
sum_g = float(0)
sum_r = float(0)
for i in range(0, 256):
    sum_b = sum_b + count_b[i]
    sum_g = sum_g + count_g[i]
    sum_r = sum_r + count_r[i]
    count_b[i] = sum_b
    count_g[i] = sum_g
    count_r[i] = sum_r
# print(count)
# 计算映射表
map_b = np.zeros(256, np.uint16)
map_g = np.zeros(256, np.uint16)
map_r = np.zeros(256, np.uint16)
for i in range(0, 256):
    map_b[i] = np.uint16(count_b[i] * 255)
    map_g[i] = np.uint16(count_g[i] * 255)
    map_r[i] = np.uint16(count_r[i] * 255)
# 映射
dst = np.zeros((height, width, 3), np.uint8)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        b = map_b[b]
        g = map_g[g]
        r = map_r[r]
        dst[i, j] = (b, g, r)
cv2.imshow('dst', dst)
cv2.waitKey(0)