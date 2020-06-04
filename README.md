# img2gcode.py

This script, `img2gcode.py` will take any image and convert it into a nice line drawing with GCode coordinates. You can take a scanned drawing, or black and white digital sketch like below and convert it into lines to be plotted.

<p align="center">
<img src=".github/sun.jpg" height=300>
</p>

You can specify the dimensions of the underlying plotter. For the example below, generated from the image above, the dimensions for [the Line-us drawing arm](https://github.com/Line-us/Line-us-Programming/blob/master/Documentation/LineUsDrawingArea.pdf)).

<p align="center">
<img src=".github/output.gif" height=300>
</p>

## Install

You need the following requirements:

- python (3.6+)
- potrace
- ffmpeg
- imagemagick

On Windows you can install with [scoop](https://scoop.sh/). Note that you must use `python37` to install Python because one of the libraries (simplification) on Windows is incompatible with Python3.8.

	scoop install potrace ffmpeg imagemagick python37

On Linux you can install with apt:
	
	sudo apt install potrace ffmpeg imagemagick python3

Once installed, you can install the required Python packages with pip:

	python3 -m pip install click loguru matplotlib numpy simplification svgpathtools svgwrite tqdm pillow

## Run

You can just download and run the script directly:

	wget https://raw.githubusercontent.com/schollz/img2gcode/master/img2gcode.py
	python3 img2gcode.py --file image.jpg --animate

After it runs it should create a folder `image.jpg.drawbot`. In that folder there are a number of files:

- `image.gc`: contains the final GCode coordinates
- `final.svg`: contains the final SVG after simplification and transforming
- `animation.mp4`: contains the animation showing the drawing process
- `potrace.svg`: shows the untransformed SVG after skeleton
- `skeleton*`: shows the skeleton transformations from the image
- `thresholded.png`: shows how the image looks after initial thresholding


## License

MIT