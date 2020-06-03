# from xml.dom import minidom

# doc = minidom.parse("test.svg")  # parseString also exists
# path_strings = [path.getAttribute('d') for path
#                 in doc.getElementsByTagName('path')]
# doc.unlink()

# path_string = " ".join(path_strings)

# print(path_string)
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from svgpathtools import svg2paths2, wsvg, Line, Path
import numpy as np
from tqdm import tqdm
from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)

import sys

#               minX maxX  minY maxY
DRAWING_AREA = [650,1775,-1000,1000]
DIMENSIONS = [1125, 2000]

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret

def fuse_linear(points,d):
	ret = []
	deleted = {}
	for _,p1 in enumerate(points):
		has_close_point = False 
		for _,p2 in enumerate(ret):
			if dist2(p1,p2) < d:
				has_close_point = True
				break
		if not has_close_point:
			ret.append(p1)
	return ret 


def cubic_bezier_sample(start, control1, control2, end):
    inputs = np.array([start, control1, control2, end])
    cubic_bezier_matrix = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  3,  0,  0],
        [ 1,  0,  0,  0]
    ])
    partial = cubic_bezier_matrix.dot(inputs)

    return (lambda t: np.array([t**3, t**2, t, 1]).dot(partial))

paths, attributes, svg_attributes = svg2paths2('out.svg')
print("have {} paths".format(len(paths)))

new_paths = []
for i,path in tqdm(enumerate(paths)):
	for j,ele in enumerate(path):
		x1 = np.real(ele.start)
		y1 = np.imag(ele.start)
		x2 = np.real(ele.end)
		y2 = np.imag(ele.end)
		if 'CubicBezier' in str(ele):
			n_segments = 8
			# get curve segment generator
			curve = cubic_bezier_sample(ele.start, ele.control1, ele.control2,ele.end)
			# get points on curve
			points = np.array([curve(t) for t in np.linspace(0, 1, n_segments)])
			for k, _ in enumerate(points):
				if k == 0:
					continue
				new_paths.append(Line(points[k-1],points[k]))
		else:
			new_paths.append(Line(ele.start,ele.end))


wsvg(new_paths,filename='output.svg')
print("wrote image to output.svg")

# deterine bounds
bounds = [1000000,-100000,100000,-1000000]
for i, ele in enumerate(new_paths):
	x1 = np.real(ele.start)
	y1 = np.imag(ele.start)
	x2 = np.real(ele.end)
	y2 = np.imag(ele.end)
	if x1 < bounds[0]:
		bounds[0] = x1 
	if x2 < bounds[0]:
		bounds[0] = x2
	if x1 > bounds[1]:
		bounds[1] = x1 
	if x2 > bounds[1]:
		bounds[1] = x2
	if y1 < bounds[2]:
		bounds[2] = y1 
	if y2 < bounds[2]:
		bounds[2] = y2
	if y1 > bounds[3]:
		bounds[3] = y1 
	if y2 > bounds[3]:
		bounds[3] = y2

global last_point
global segmenti

# transform points
print(bounds)
last_point = [0,0]
segments = []
segment = []
for i, ele in enumerate(new_paths):
	x1 = (np.real(ele.start) - bounds[0]) / (bounds[1]-bounds[0]) * (DRAWING_AREA[1]-DRAWING_AREA[0])+DRAWING_AREA[0]
	y1 = (np.imag(ele.start) - bounds[2]) / (bounds[3]-bounds[2]) * (DRAWING_AREA[3]-DRAWING_AREA[2])+DRAWING_AREA[2]
	x2 = (np.real(ele.end) - bounds[0]) / (bounds[1]-bounds[0]) * (DRAWING_AREA[1]-DRAWING_AREA[0])+DRAWING_AREA[0]
	y2 = (np.imag(ele.end) - bounds[2]) / (bounds[3]-bounds[2]) * (DRAWING_AREA[3]-DRAWING_AREA[2])+DRAWING_AREA[2]
	x1 = round(x1,1)
	y1 = round(y1,1)
	x2 = round(x2,1)
	y2 = round(y2,1)

	d = dist2([x1,y1],[x2,y2])
	if x1 != last_point[0] and y1 != last_point[1] and last_point[0] !=0 and last_point[1] !=0:
		segments.append(segment)
		segment = []
	elif d > 100000: # this is to prevent the border
		segments.append(segment)
		segment = []
		continue			
	segment.append(Line(complex(x1,y1),complex(x2,y2)))
	last_point = [x2,y2]
segments.append(segment)

bounds = DRAWING_AREA

print("have {} segments".format(len(segments)))
print("first segment has {} lines".format(len(segments[0])))
total_points_original = 0
total_points_new = 0
new_new_paths = []
new_new_paths_flat = []
for i,segment in enumerate(segments):
	coords = []
	for j,ele in enumerate(segment):
		x1 = np.real(ele.start)
		y1 = np.imag(ele.start)
		x2 = np.real(ele.end)
		y2 = np.imag(ele.end)
		if j == 0:
			coords.append([x1,y1])
		coords.append([x2,y2])
	total_points_original += len(coords)
	simplified = coords
	# simplified = fuse_linear(simplified,30)
	simplified = simplify_coords(simplified, 5.0)
	if len(simplified) < 2:
		continue
	total_points_new += len(simplified)
	paths = Path()
	for j,coord in enumerate(simplified):
		if j==0:
			continue
		paths.append(Line(complex(simplified[j-1][0],simplified[j-1][1]),complex(simplified[j][0],simplified[j][1])))
		new_new_paths_flat.append(Line(complex(simplified[j-1][0],simplified[j-1][1]),complex(simplified[j][0],simplified[j][1])))
	if i > -1:
		new_new_paths.append(paths)

print("had {} points ".format(total_points_original))
print("now have {} points".format(total_points_new))
print("now have {} lines".format(len(new_new_paths)))
wsvg(new_new_paths,filename='output2.svg')
print("wrote image to output2.svg")

print(bounds)

fig, ax = plt.subplots()
ax.set_aspect(aspect=1)
plt.axis(bounds)

print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

t = tqdm(total=len(new_new_paths_flat)) 

last_point = [0,0]
segmenti = 0
colors = list(mcolors.TABLEAU_COLORS)

def update(i):
	if i > len(new_new_paths_flat):
		return
	global last_point, segmenti
	t.update(1)
	label = 'timestep {0}'.format(i)
	ele = new_new_paths_flat[i]
	x1 = np.real(ele.start)
	y1 = np.imag(ele.start)
	x2 = np.real(ele.end)
	y2 = np.imag(ele.end)
	if x1 != last_point[0] and y1 != last_point[1]:
		segmenti = segmenti + 1
	d = dist2([x1,y1],[x2,y2])
	plt.plot([x1,x2],[y1,y2],'-',color=colors[segmenti % len(colors)])
	last_point = [x2,y2]
	return



anim = FuncAnimation(fig, update, frames=len(new_new_paths_flat), interval=50, repeat=False)
plt.show()
# print('saving animation')
# anim.save('line.mp4', dpi=300, writer='ffmpeg')
# print("wrote draw pattern to line.gif")




