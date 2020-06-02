from PIL import Image
import numpy as np
from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)


import sys
import time

im = np.array(Image.open('out.png'))

pixels = np.transpose(np.nonzero(im==False))
pixels_todo = []
for _, pixel in enumerate(pixels):
	pixels_todo.append(hash(frozenset(pixel)))
pixels_done = []
print("found {} pixels".format(len(pixels)))

segments = []
while len(pixels_done) < len(pixels):
	segment = []
	next_pixel = []
	for _, pixel in enumerate(pixels):
		if hash(frozenset(pixel)) in pixels_done:
			continue
		next_pixel = pixel
		break
	while len(next_pixel) > 0:
		p = next_pixel
		next_pixel = []
		segment.append(p)
		pixels_done.append(hash(frozenset(p)))
		direction_tries = 0
		direction = 0
		while direction_tries < 4:
			if direction == 0:
				ps = [ [p[0]+1,p[1]],[p[0]+1,p[1]+1],[p[0]+1,p[1]-1] ]
			elif direction == 1:
				ps =[ [p[0],p[1]-1],[p[0]+1,p[1]-1],[p[0]-1,p[1]-1] ]
			elif direction == 2:
				ps =[ [p[0]-1,p[1]],[p[0]-1,p[1]-1],[p[0]-1,p[1]+1] ]
			elif direction == 3:
				ps =[ [p[0],p[1]+1],[p[0]-1,p[1]+1],[p[0]+1,p[1]+1] ]

			found_pixel = False
			for _, p2 in enumerate(ps):
				if hash(frozenset(p2)) in pixels_done:
					continue
				if hash(frozenset(p2)) in pixels_todo:
					next_pixel = p2
					found_pixel = True
					break

			if not found_pixel:
				direction = (direction + 1) % 4
				direction_tries += 1
			else:
				break

	if len(segment) == 0:
		break
	segments.append(segment)
	

dstring = ""
for _, segment in enumerate(segments):
	simplified = simplify_coords(segment, 1.0)
	if len(simplified) <2:
		continue
	if hash(frozenset(simplified[0]))==hash(frozenset(simplified[1])):
		continue
	for i,s in enumerate(simplified):
		if i == 0:
			dstring += "M "
		else:
			dstring += "L "
		dstring += "{},{} ".format(s[0],s[1])

print(dstring)