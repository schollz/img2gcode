# from xml.dom import minidom

# doc = minidom.parse("test.svg")  # parseString also exists
# path_strings = [path.getAttribute('d') for path
#                 in doc.getElementsByTagName('path')]
# doc.unlink()

# path_string = " ".join(path_strings)

# print(path_string)
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# import matplotlib.colors as mcolors
# from matplotlib.animation import FuncAnimation
from svgpathtools import svg2paths2, wsvg, Line, Path
from svg.path import parse_path
import numpy as np
from tqdm import tqdm
from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)
from loguru import logger as log
import click
from PIL import Image, ImageDraw

import hashlib
import ntpath
import os
import subprocess
import sys
from shutil import copyfile


def dist2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


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
            for j in range(i + 1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret


def fuse_linear(points, d):
    ret = []
    deleted = {}
    for _, p1 in enumerate(points):
        has_close_point = False
        for j, p2 in enumerate(ret):
            if dist2(p1, p2) < d:
                has_close_point = True
                break
        if not has_close_point:
            ret.append(p1)
    return ret


def cubic_bezier_sample(start, control1, control2, end):
    inputs = np.array([start, control1, control2, end])
    cubic_bezier_matrix = np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
    )
    partial = cubic_bezier_matrix.dot(inputs)

    return lambda t: np.array([t ** 3, t ** 2, t, 1]).dot(partial)


def processSVG(fnamein, fnameout, simplifylevel=5, pruneLittle=7, drawing_area=[650, 1775, -1000, 1000]):
    paths, attributes, svg_attributes = svg2paths2(fnamein)
    log.info("have {} paths", len(paths))

    log.debug("converting beziers to lines")

    new_paths = []
    bounds = [1000000, -100000, 100000, -1000000]
    for ii, path in enumerate(attributes):
        new_path = []
        point_start = []
        point_end = []
        path = parse_path(attributes[ii]['d'])
        for jj, ele in enumerate(path):
            x1 = np.real(ele.start)
            y1 = np.imag(ele.start)
            x2 = np.real(ele.end)
            y2 = np.imag(ele.end)
            if jj == 0:
                point_start = [x1,y1]
            point_end = [x2,y2]
            if ii == 0:
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
            if "CubicBezier" in str(ele):
                n_segments = 2
                # get curve segment generator
                curve = cubic_bezier_sample(
                    ele.start, ele.control1, ele.control2, ele.end
                )
                # get points on curve
                points = np.array([curve(t) for t in np.linspace(0, 1, n_segments)])
                for k, _ in enumerate(points):
                    if k == 0:
                        continue
                    new_path.append(Line(complex(np.real(points[k - 1]),np.imag(points[k-1])),complex(np.real(points[k ]),np.imag(points[k]))))
            elif "Line" in str(ele):
                new_path.append(Line(ele.start, ele.end))
            elif "Move" in str(ele):
                point_start = [x1,y1]
            elif "Close" in str(ele):
                # if dist2(point_start,point_end)<1:
                #     new_path = new_path[:int(len(new_path)/1.5)]
                new_paths.append(new_path)
                new_path =[]

        if len(new_path) > 0:
            # if dist2(point_start,point_end)<1:
            #     new_path = new_path[:int(len(new_path)/1.5)]
            new_paths.append(new_path)


    #transform points
    print(bounds)
    bounds = [0.0, 10*(drawing_area[1]-drawing_area[0]), 0.0,10*(drawing_area[3]-drawing_area[2])]
    num_coords = 0
    num_coords_simplified = 0
    new_paths_flat = []
    new_new_paths = []
    for j, path in enumerate(new_paths):
        coords = []
        for i, ele in enumerate(path):
            x1 = (np.real(ele.start) - bounds[0]) / (bounds[1] - bounds[0]) * (
                drawing_area[1] - drawing_area[0]
            ) + drawing_area[0]
            y1 = (np.imag(ele.start) - bounds[2]) / (bounds[3] - bounds[2]) * (
                drawing_area[3] - drawing_area[2]
            ) + drawing_area[2]
            x2 = (np.real(ele.end) - bounds[0]) / (bounds[1] - bounds[0]) * (
                drawing_area[1] - drawing_area[0]
            ) + drawing_area[0]
            y2 = (np.imag(ele.end) - bounds[2]) / (bounds[3] - bounds[2]) * (
                drawing_area[3] - drawing_area[2]
            ) + drawing_area[2]
            x1 = round(x1)
            y1 = round(y1)
            x2 = round(x2)
            y2 = round(y2)
            coords.append([x1,y1])
            coords.append([x2,y2])

        simplified = coords
        simplified = simplify_coords(simplified ,simplifylevel)
        
        num_coords += len(coords)
        num_coords_simplified += len(simplified)

        new_path = []
        for i,coord in enumerate(simplified):
            
            if i == 0:
                continue
            path = Line(complex(simplified[i-1][0],simplified[i-1][1]),complex(simplified[i][0],simplified[i][1]))
            new_path.append(path)
            new_paths_flat.append(path)
            # new_paths[j][i] = Line(complex(x1,y1),complex(x2,y2))
        new_new_paths.append(new_path)
    bounds = drawing_area

    log.debug(f"have {num_coords} coordinates")
    log.debug(f"have {num_coords_simplified} coordinates after simplifying")
#     with open("test.svg","w") as f:
#         f.write("""<?xml version="1.0" ?><svg baseProfile="full" height="600px" version="1.1" viewBox="940.1115 425.9115 510.777 933.177" width="329px" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink"><defs/>
# """)
#         for i, path in enumerate(new_new_paths):
#             pathstring = ""
#             for j, ele in enumerate(path):
#                 x1 = np.real(ele.start)
#                 y1 = np.imag(ele.start)
#                 x2 = np.real(ele.end)
#                 y2 = np.imag(ele.end)
#                 if j == 0:
#                     pathstring += f"M {int(x1)},{int(y1)} "
                
#                 if j > 0 or len(path) == 1:
#                     pathstring += f"L {int(x2)},{int(y2)} "
#             f.write(f"<path d=\"{pathstring}\""+""" fill="none" stroke="#000000" stroke-width="0.777"/>""" + "\n")
#         f.write("</svg>\n")
    wsvg(new_paths_flat, filename=fnameout)
    log.debug("wrote image to {}", fnameout)

    log.info(bounds)

    gcodestring = "G01 Z1000"
    for i, path in enumerate(new_new_paths):
        coords = []
        for j, ele in enumerate(path):
            x1 = np.real(ele.start)
            y1 = np.imag(ele.start)
            x2 = np.real(ele.end)
            y2 = np.imag(ele.end)
            if j == 0:
                gcodestring += f"\nG01 X{int(x1)} Y{int(y1)} Z1000"
                gcodestring += f"\nG01 Z0"
            else:
                gcodestring += f"\nG01 X{int(x1)} Y{int(y1)} Z0"    

        gcodestring += "\nG01 Z1000"
    with open("image.gc","w")as f:
        f.write(gcodestring.strip())

    return new_paths_flat, bounds


def animateProcess(new_new_paths_flat, bounds, fname="out.gif"):
    images = []
    color_1 = (0, 0, 0)
    color_2 = (255, 255, 255)
    print(bounds)
    im = Image.new('RGB', (bounds[1]-bounds[0], bounds[3]-bounds[2]), color_2)
    last_point = [0,0]
    gifmod = 4
    if len(new_new_paths_flat) > 50:
        gifmod = int(len(new_new_paths_flat)/50)
    for i,ele in enumerate(new_new_paths_flat):
        x1 = np.real(ele.start)-bounds[0]
        y1 = np.imag(ele.start)-bounds[2]
        x2 = np.real(ele.end)-bounds[0]
        y2 = np.imag(ele.end)-bounds[2]
        if x1 != last_point[0] and y1 != last_point[1]:
            pass # change color
        draw = ImageDraw.Draw(im)
        draw.line((x1,y1,x2,y2), fill=color_1,width=6)
        if i%gifmod == 0:
            im0 = im.copy()
            images.append(im0)
        last_point = [x2, y2]
    log.debug(len(images))
    log.debug(f"saving {fname}")
    images[0].save(fname,
               save_all=True, append_images=images[1:], optimize=False, duration=1, loop=2)
    # global last_point, segmenti
    # if fname != "":
    #     import matplotlib

    #     matplotlib.use("Agg")

    # fig, ax = plt.subplots()
    # ax.set_aspect(aspect=1)
    # plt.axis(bounds)

    # print(
    #     "fig size: {0} DPI, size in inches {1}".format(
    #         fig.get_dpi(), fig.get_size_inches()
    #     )
    # )

    # t = tqdm(total=len(new_new_paths_flat))

    # last_point = [0, 0]
    # segmenti = 0
    # colors = list(mcolors.TABLEAU_COLORS)

    # def update(i):
    #     global last_point, segmenti
    #     if i > len(new_new_paths_flat):
    #         return
    #     t.update(1)
    #     ele = new_new_paths_flat[i]
    #     x1 = np.real(ele.start)
    #     y1 = np.imag(ele.start)
    #     x2 = np.real(ele.end)
    #     y2 = np.imag(ele.end)
    #     if x1 != last_point[0] and y1 != last_point[1]:
    #         segmenti = segmenti + 1
    #     plt.plot(
    #         [x1, x2], [y1, y2], "-", color=colors[segmenti % len(colors)], linewidth=0.4
    #     )
    #     last_point = [x2, y2]
    #     return

    # anim = FuncAnimation(
    #     fig, update, frames=len(new_new_paths_flat), interval=50, repeat=False
    # )
    # if fname != "":
    #     log.debug("saving animation")
    #     anim.save(fname, dpi=300, writer="ffmpeg")
    # else:
    #     plt.show()


@click.command()
@click.option("--file", prompt="image in?", help="svg to process")
@click.option("--folder",default=".", help="folder to output into")
@click.option("--animate/--no-animate", default=False)
@click.option("--overwrite/--no-overwrite", default=True)
@click.option("--centerline/--no-centerline", default=False)
@click.option("--minx", default=650, help="minimum x")
@click.option("--maxx", default=1775, help="maximum x")
@click.option("--miny", default=-1000, help="minimum y")
@click.option("--maxy", default=1000, help="maximum y")
@click.option("--maxy", default=1000, help="maximum y")
@click.option("--prune", default=7, help="amount of pruning of small things")
@click.option("--simplify", default=5, help="simplify level")
@click.option("--threshold", default=60, help="percent threshold (0-100)")
def run(folder, prune,centerline, file, simplify, overwrite, animate, minx, maxx, miny, maxy, threshold):
    imconvert = "convert"
    if os.name == "nt":
        imconvert = "imconvert"

    if folder != ".":
        try:
            os.mkdir(folder)
        except:
            pass

    foldername = os.path.join(folder, ntpath.basename(file) + ".img2gcode")
    try:
        os.mkdir(foldername)
    except:
        pass

    copyfile(file, os.path.join(foldername, ntpath.basename(file)))

    log.info(f"working in {foldername}")
    os.chdir(foldername)
    file = ntpath.basename(file)

    width = maxy - miny
    height = maxx - minx
    if not os.path.exists("potrace.svg") or overwrite:
        if centerline:
            cmd = f"{imconvert} {file} -resize {width}x{height} -background White -gravity center -extent {width}x{height} -threshold {threshold}%% thresholded.png"
            log.debug(cmd)
            subprocess.run(cmd.split())

            cmd = f"{imconvert} thresholded.png -negate -morphology Thinning:-1 Skeleton skeleton.png"
            log.debug(cmd)
            subprocess.run(cmd.split())

            cmd = f"{imconvert} skeleton.png -negate skeleton_negate.png"
            log.debug(cmd)
            subprocess.run(cmd.split())

            cmd = f"{imconvert} skeleton_negate.png -rotate 90 skeleton_border.png"
            log.debug(cmd)
            subprocess.run(cmd.split())

            cmd = f"{imconvert} skeleton_border.png -flip skeleton_border_flip.bmp"
            log.debug(cmd)
            subprocess.run(cmd.split())

            cmd = f"potrace -b svg -o potrace.svg skeleton_border_flip.bmp"
            log.debug(cmd)
            subprocess.run(cmd.split())
        else:
            cmd = f"{imconvert} {file} -resize {width}x{height} -background White -gravity center -extent {width}x{height} -threshold {threshold}%% thresholded.png"
            log.debug(cmd)
            subprocess.run(cmd.split())

            cmd = f"{imconvert} thresholded.png -rotate 90 -flip  thresholded.bmp"
            log.debug(cmd)
            subprocess.run(cmd.split())

            cmd = f"potrace -b svg -o potrace.svg -n thresholded.bmp"
            log.debug(cmd)
            subprocess.run(cmd.split())
            os.remove("thresholded.bmp")

    new_new_paths_flat, bounds = processSVG("potrace.svg", "final.svg",simplifylevel=simplify,pruneLittle=prune,drawing_area = [minx,maxx,miny,maxy])

    cmd = f"{imconvert} final.svg -rotate 270 final.png"
    log.debug(cmd)
    subprocess.run(cmd.split())

    animatefile = ""
    if animate:
        animatefile = "1.gif"
        animateProcess(new_new_paths_flat, bounds, animatefile)
        cmd = f"{imconvert} 1.gif -rotate 270 animation.gif"
        log.debug(cmd)
        subprocess.run(cmd.split())
        os.remove("1.gif")

if __name__ == "__main__":
    run()
