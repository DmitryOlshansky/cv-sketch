#!/usr/bin/env python
import cv2
import random
import numpy as np
from math import sqrt, pow

def circle_for(x1, y1, x2, y2, x3, y3):
	ynorm =  (y2*y2 - y3*y3)*(x2 - x1)/2 + (x3-x2)*(y2*y2-y1*y1)/2 + (x1 - x3)*(y2 - y3)*(x2- x1) 
	y = ynorm / ((y2 - y3)*(x2 - x1) + (x3-x2)*(y2-y1))
	x = (y1 - y2)*(2*y - y1 - y2)/(x2 - x1)/2 + (x1+x2)/2
	R = sqrt(pow(x-x1,2) + pow(y-y1,2))
	return (x,y,R)

print circle_for(1.0, 0.0, 0.0, 1.0, sqrt(2)/2, sqrt(2)/2)


img = cv2.imread('hv.jpg', 0)

img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

height, width = img.shape[:2]
maxRadius = 500

img = cv2.Canny(img, 80, 180)
it = np.nditer(img, flags=['multi_index'])
points = []
while not it.finished:
	if it[0] > 0:
		points.append((float(it.multi_index[1]), float(it.multi_index[0])))
	it.iternext()

def randomized_hough(points, samples, threshold):
	buffer = {}
	for i in xrange(samples):
		sample = random.sample(points, 3)
		try:
			x,y,R = circle_for(sample[0][0], sample[0][1], sample[1][0], sample[1][1],
				sample[2][0], sample[2][1])
			if x > 0 and x < width and y > 0 and y < height and R > 5 and R < maxRadius:
				tripple = int(x)/2, int(y)/2, int(R)/4
				if tripple in buffer:
					buffer[tripple] += 1
				else:
					buffer[tripple] = 1
		except ZeroDivisionError:
			pass
	for (k,v) in buffer.items():
		if v > threshold:
			yield k[0]*2, k[1]*2, k[2]*4
print height, width
img2 = np.zeros((height, width, 3), np.uint8)

for (x,y,R) in randomized_hough(points, 200000, 3):
	cv2.circle(img2, (x, y), R, (0,0,255), 1)

cv2.imshow('Hough transform', img)
cv2.waitKey(0)
cv2.imshow('Hough transform 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
