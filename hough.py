#!/usr/bin/env python
import cv2
import numpy as np
import random

original = cv2.imread('img2.jpg', 3)
img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
img = cv2.blur(img, (3,3))
img = cv2.Canny(img, 120, 200)

def draw_fit(x, y):
	m,b = ls_fit_line(img, x, y)
	cv2.line(original, (0, int(b)), (500, int(m*500+b)), (255,0,0))	

def draw_line(m, b):
	cv2.line(original, (0, int(b)), (500, int(m*500+b)), (255,0,0))	

# least-squares fit line
def ls_fit_line(img, x, y):
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

class Accumulator:
	
	def __init__(self):
		self.thrd = 2
		self.epouchs = 10
		self.iters = 100
		self.sample_size = 2

	def key_for(self, params):
		return None

	def fit_curve(self, img, sample):
		return None

	def on_curve(self, curve):
		return False

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

	def best_curve(self):
		maxW = 0
		curve = None
		for v in self.accum.values():
			if v[1] > maxW:
				maxW = v[1]
				curve = v[0]
		if maxW > self.thrd:
			return curve
		else:
			return None

	# generator over all detected curves
	def __call__(self, img, points=None):
		# Get points
		self.accum = {}
		if points != None:
			self.points = points
		else:
			self.points = []
			it = np.nditer(img, flags=["multi_index"])
			while not it.finished:
				if it[0] > 0:
					self.points.append((it.multi_index[1], it.multi_index[0]))
				it.iternext()
		# iterate across epouchs
		for i in xrange(self.epouchs):
			self.accum = {}
			for j in xrange(self.iters):
				if len(self.points) < self.sample_size:
					break
				sample = random.sample(self.points, self.sample_size)
				curve = self.fit_curve(img, sample)
				if curve: # None if failed to fit
					self.accumulate(curve)
			curve = self.best_curve()
			if curve != None:
				new_points = filter(lambda p: not self.on_curve(curve, p), self.points)
				# test for sufficient amount of removed points
				print curve, len(self.points) - len(new_points)
				if len(self.points) - len(new_points) > self.min_curve:
					yield curve
					self.points = new_points
		raise StopIteration


class EllipseAccum(Accumulator):

	def __init__(self, epouchs=10, iters=100, key_tolerance=4, curve_tolerance=0.01, threshold=2, min_curve=12):
		self.accum = {}
		self.thrd = threshold
		self.min_curve = min_curve
		self.epouchs = epouchs
		self.iters = iters
		self.sample_size = 3
		self.tol = key_tolerance
		self.on_curve_tol = curve_tolerance

	def key_for(self, params):
		return int(params[0])/self.tol, int(params[1])/self.tol, int(params[2])/self.tol,int(params[3])/self.tol

	def fit_curve(self, img, pts):
		# one step of RHT algortihm, given 3 random points in pts
		x1,y1 = pts[0][0], pts[0][1]
		x2,y2 = pts[1][0], pts[1][1]
		x3,y3 = pts[2][0], pts[2][1]
		try:
			m1, b1 = ls_fit_line(img, x1, y1)
			m2, b2 = ls_fit_line(img, x2, y2)
			m3, b3 = ls_fit_line(img, x3, y3)
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
				
				theta = 0.5*np.arctan(2*b/(a-c))
				tgTheta2 = np.tan(theta)
				tgTheta2 *= tgTheta2
				# print tgTheta2
				r1 = np.sqrt(abs((1 - tgTheta2)/(a - c*tgTheta2)))
				r2 = np.sqrt(abs((1 - tgTheta2)/(c - a*tgTheta2)))
				if r1 == float('inf') or r2 == float('inf'):
					return None
				# cv2.ellipse(original, (int(x),int(y)), (int(r1), int(r2)), theta*180/np.pi, 0, 360, (0, 0, 160))
				return (x,y, r1, r2, theta*180/np.pi)
			else:
				return None
		except ZeroDivisionError:
			return None
		except np.linalg.linalg.LinAlgError:
			return None

	def on_curve(self, curve, p):
		x,y = p
		x0,y0,a,b,theta = curve
		# TODO: account for theta
		x -= x0
		y -= y0
		th = theta*np.pi/180
		x,y = x*np.cos(th) + y*np.sin(th), -x*np.sin(th) + y*np.cos(th)
		if abs(x*x/(a*a) + y*y/(b*b) - 1.0) < self.on_curve_tol:
			return True
		return False

class CirclesAccum(Accumulator):

	def __init__(self, epouchs=10, iters=100, key_tolerance=2, curve_tolerance=4, threshold=2, min_curve=12):
		self.thrd = threshold
		self.sample_size = 3
		self.min_curve = min_curve
		self.tol = key_tolerance
		self.epouchs = epouchs
		self.iters = iters
		self.on_curve_tol = curve_tolerance

	def key_for(self, params):
		return int(params[0])/self.tol, int(params[1])/self.tol, int(params[2])/self.tol

	def fit_curve(self, img, sample):
		try:
			x1,y1 = sample[0]
			x2,y2 = sample[1]
			x3,y3 = sample[2]
			ynorm =  (y2*y2 - y3*y3)*(x2 - x1)/2 + (x3-x2)*(y2*y2-y1*y1)/2 + (x1 - x3)*(y2 - y3)*(x2- x1) 
			y = ynorm / ((y2 - y3)*(x2 - x1) + (x3-x2)*(y2-y1))
			x = (y1 - y2)*(2*y - y1 - y2)/(x2 - x1)/2 + (x1+x2)/2
			R = np.sqrt(pow(x-x1,2) + pow(y-y1,2))
			return (x,y,R)
		except ZeroDivisionError:
			return None

	def on_curve(self, curve, point):
		x,y = point
		x0,y0,R = curve
		if abs((x-x0)*(x-x0) + (y-y0)*(y-y0) - R*R) < self.on_curve_tol*self.on_curve_tol:
			return True
		return False

class LinesAccum(Accumulator):

	def __init__(self, epouchs=10, iters=100, key_tolerance=1, curve_tolerance=1, threshold=2, min_curve=12):
		self.thrd = threshold
		self.sample_size = 2
		self.min_curve = min_curve
		self.tol = key_tolerance
		self.epouchs = epouchs
		self.iters = iters
		self.on_curve_tol = curve_tolerance

	def key_for(self, params):
		if params[0] == float('inf'):
			return params[1]
		else:
			return int(np.arctan(params[0])*180/np.pi)/self.tol, int(params[1])

	def fit_curve(self, img, sample):
		x1,y1 = sample[0]
		x2,y2 = sample[1]
		if x1 != x2: 
			m = float(y1 - y2)/(x1 - x2)
			b = y1 - x1 * m
		else:
			m = float('inf')
			b = x1
		return m,b

	def on_curve(self, curve, point):
		x,y = point
		m,b = curve
		if m == float('inf') and abs(b - x) < self.on_curve_tol:
			return True
		elif abs(x*m+b - y) < self.on_curve_tol:
			return True
		return False

lines = LinesAccum(epouchs=45, iters=250, key_tolerance=5, curve_tolerance=1, min_curve=20)
for m,b in lines(img):
	if m != float('inf'):
		cv2.line(original, (int(0), int(b)), (int(800), int(800*m+b)), (0, 255, 0))
	else:
		cv2.line(original, (int(b), 0), (int(b), int(600)), (0, 255, 0))

circles = CirclesAccum(epouchs=10, iters=10000, key_tolerance=5, min_curve=10)
for x,y,r in circles(img, lines.points):
	cv2.circle(original, (int(x),int(y)), int(r), (255, 0, 0))

ellipses = EllipseAccum(epouchs=8, iters=6000, key_tolerance=5, curve_tolerance=0.05, min_curve=10)
# Use points filtered for lines
for x,y,r1,r2,theta in ellipses(img, lines.points):
	cv2.ellipse(original, (int(x),int(y)), (int(r1), int(r2)), theta, 0, 360, (0, 0, 255))

#cv2.imshow('After canny + filter', img)
#cv2.waitKey(0)
cv2.imshow('Some image', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
