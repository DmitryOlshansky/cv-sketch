#!/usr/bin/env python
import cv2
import numpy as np
import random

original = cv2.imread('img.jpg', 3)
img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
img = cv2.blur(img, (3,3))
img = cv2.Canny(img, 120, 200)

def draw_fit(x, y):
	m,b = fit_line(img, x, y)
	cv2.line(original, (0, int(b)), (500, int(m*500+b)), (255,0,0))	

def draw_line(m, b):
	cv2.line(original, (0, int(b)), (500, int(m*500+b)), (255,0,0))	


def fit_line(img, x, y):
	height, width = img.shape[0:2]
	sx = -3+x if x >= 3 else 0
	sy = -3+y if y >= 3 else 0
	ex = 4+x if x < width-4 else width
	ey = 4+y if y < height-4 else height
	neib = img[sy:ey, sx:ex]
	pts = []
	it = np.nditer(neib, flags=["multi_index"])
	while not it.finished:
		if it[0] > 0:
			pts.append((it.multi_index[1]+sx, it.multi_index[0]+sy))
		it.iternext()
	cx = sum(map(lambda v: v[0], pts))/float(len(pts))
	cy = sum(map(lambda v: v[1], pts))/float(len(pts))
	nom = sum(map(lambda v: (v[0] - cx)*(v[1] - cy), pts))
	denom = sum(map(lambda v: (v[0] - cx) * (v[0] - cx), pts))
	if denom == 0.0:
		m = 1e20
	else:
		m = nom/denom
	b = cy - m*cx
	return (m, b)

def intersection(line1, line2):
	b1 = line1[1]
	m1 = line1[0]
	b2 = line2[1]
	m2 = line2[0]

	x = (b2 - b1)/(m1 - m2)
	y = b1 + m1*(b2 - b1)/(m1 - m2)
	return (x,y)

def line_by_points(m, t):
	mx,my = m[0], m[1]
	tx,ty = t[0], t[1]
	slope = (my - ty)/(mx - tx)
	b = -slope*tx + ty
	return (slope, b)

# one step of RHT algortihm, given 3 random points in pts
def rht_ellipse_step(img, pts):
	x1,y1 = pts[0][0], pts[0][1]
	x2,y2 = pts[1][0], pts[1][1]
	x3,y3 = pts[2][0], pts[2][1]
	try:
		m1, b1 = fit_line(img, x1, y1)
		m2, b2 = fit_line(img, x2, y2)
		m3, b3 = fit_line(img, x3, y3)

		t12 = intersection((m1, b1), (m2, b2))
		t23 = intersection((m2, b2), (m3, b3))
		m12x = (x1 + x2)/2
		m12y = (y1 + y2)/2
		m23x = (x2 + x3)/2
		m23y = (y2 + y3)/2

		bisector12 = line_by_points(t12, (m12x, m12y))
		bisector23 = line_by_points(t23, (m23x, m23y))
		#draw_line(*bisector12)
		#draw_line(*bisector23)
		# x,y - center of ellipse
		x,y = intersection(bisector23, bisector12)
		#cv2.circle(original, (int(x), int(y)), 15, (0, 255, 90))
		x1 -= x
		x2 -= x
		x3 -= x
		y1 -= y
		y2 -= y
		y3 -= y
		# x1..x3, y1..y3 translated to the origin of ellipse
		a, b, c = np.linalg.solve([
			[x1*x1, 2*x1*y1, y1*y1], 
			[x2*x2, 2*x2*y2, y2*y2], 
			[x3*x3, 2*x3*y3, y3*y3]
			], [1, 1, 1])
		if 4*a*c > b*b:
			#r1 = np.sqrt(1.0/a)
			#r2 = np.sqrt(1.0/c)
			
			theta = 0.5*np.arctan(b/(a-c))
			tgTheta = np.tan(theta)
			r1 = 1/np.sqrt(abs((a - c*tgTheta)/(1 - tgTheta*tgTheta)))
			r2 = 1/np.sqrt(abs(a + c - 1/r1/r1))
			# cv2.ellipse(original, (int(x),int(y)), (int(r1), int(r2)), theta*180/np.pi, 0, 360, (0, 0, 160))
			return (x,y, r1, r2, theta*180/np.pi)
		else:
			return None
	except ZeroDivisionError:
		return None
	except np.linalg.linalg.LinAlgError:
		return None

def rht_ellipses(img, steps, tolerance=4):
	accum = EllipseAccum(tolerance)
	
	points = []
	it = np.nditer(img, flags=["multi_index"])
	while not it.finished:
		if it[0] > 0:
			points.append((it.multi_index[1], it.multi_index[0]))
		it.iternext()
	for i in xrange(steps):
		sample = random.sample(points, 3)
		e = rht_ellipse_step(img, sample)
		if e:
			accum.accumulate(e)

	for v in accum.found():
		x,y,r1,r2,theta = v
		cv2.ellipse(original, (int(x),int(y)), (int(r1), int(r2)), 0, 0, 360, (0, 0, 255))
		print v

def rht_lines_step(sample):
	x1,y1 = sample[0]
	x2,y2 = sample[1]
	if x1 != x2: 
		m = (y1 - y2)/(x1 - x2)
		b = y1 - x1 * m
	else:
		m = float('inf')
		b = x1
	return m,b

def rht_lines(img, steps, tolerance=4):
	accum = LinesAccum(tolerance)
	points = []
	it = np.nditer(img, flags=["multi_index"])
	while not it.finished:
		if it[0] > 0:
			points.append((it.multi_index[1], it.multi_index[0]))
		it.iternext()
	for i in xrange(steps):
		sample = random.sample(points, 2)
		e = rht_lines_step(sample)
		if e:
			accum.accumulate(e)
	for v in accum.found():
		m,b = v
		if m != float('inf'):
			cv2.line(original, (int(0), int(b)), (int(800), int(800*m+b)), (0, 255, 0))
		else:
			cv2.line(original, (int(b), 0), (int(b), int(600)), (0, 255, 0))

class Accumulator:
	
	def __init__(self):
		self.accum = {}
		self.thrd = 2

	def key_for(self, params):
		return None

	def accumulate(self, params):
		key = self.key_for(params)
		if key in self.accum:
			prior = self.accum[key][0]
			w = self.accum[key][1]
			arr = np.array(params)
			weighted = (prior*w + arr)/(w+1)
			kw = self.key_for(weighted)
			if kw != key:
				del self.accum[key]
			self.accum[kw] = (weighted, w+1)
		else:
			self.accum[key] = (np.array(params), 1)

	def found(self):
		for v in self.accum.values():
			if v[1] > self.thrd:
				yield v[0]

class EllipseAccum(Accumulator):

	def __init__(self, tolerance):
		self.accum = {}
		self.thrd = 2
		self.tol = tolerance

	def key_for(self, params):
		return int(params[0])/self.tol, int(params[1])/self.tol, int(params[2])/self.tol,int(params[3])/self.tol

class LinesAccum(Accumulator):

	def __init__(self, tolerance):
		self.accum = {}
		self.thrd = 10
		self.tol = tolerance

	def key_for(self, params):
		if params[0] == float('inf'):
			return int(params[1])/self.tol
		else:
			return int(params[0])/self.tol, int(params[1])/self.tol

#rht_ellipses(img, 2000)
rht_lines(img, 2000)
#cv2.imshow('After canny + filter', img)
#cv2.waitKey(0)
cv2.imshow('Some image', original)
cv2.waitKey(0)
cv2.destroyAllWindows()