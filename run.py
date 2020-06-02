# from xml.dom import minidom

# doc = minidom.parse("test.svg")  # parseString also exists
# path_strings = [path.getAttribute('d') for path
#                 in doc.getElementsByTagName('path')]
# doc.unlink()

# path_string = " ".join(path_strings)

# print(path_string)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation
from svgpathtools import svg2paths2, wsvg, Line
import numpy as np
from tqdm import tqdm

import sys

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

bounds = [1000,-1000,1000,-1000]
paths, attributes, svg_attributes = svg2paths2('test.svg')
print("have {} paths".format(len(paths)))

new_paths = []
for i,path in enumerate(paths):
	for j,ele in enumerate(path):
		x1 = np.real(ele.start)
		y1 = np.imag(ele.start)
		x2 = np.real(ele.end)
		y2 = np.imag(ele.end)
		if 'CubicBezier' in str(ele):
			n_segments = 2
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


print("wrote image to output.svg")
wsvg(new_paths,filename='output.svg')
paths, attributes, svg_attributes = svg2paths2('output.svg')
print("have {} paths".format(len(paths)))

print(bounds)

fig, ax = plt.subplots()
fig.set_tight_layout(True)
plt.axis(bounds)

print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

t = tqdm(total=len(new_paths)) 
def update(i):
	t.update(1)
	label = 'timestep {0}'.format(i)
	ele = new_paths[i]
	x1 = np.real(ele.start)
	y1 = np.imag(ele.start)
	x2 = np.real(ele.end)
	y2 = np.imag(ele.end)
	plt.plot([x1,x2],[y1,y2],'k-')
	return _, ax



anim = FuncAnimation(fig, update, frames=np.arange(0, len(new_paths)), interval=1)
plt.show()
# print('saving animation')
# anim.save('line.gif', dpi=80, writer='imagemagick')
# print("wrote draw pattern to line.gif")


