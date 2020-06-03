# drawbot

drawbot is a python script that utilizes common tools to turn a drawing (scanned or from a tablet) into a svg line drawing whose coordinates are compatabile with the drawing area of a drawing robot arm.

Basically, you can take a drawing, like this:

<p align="center">
<img src=".github/sun.jpg" height=300>
</p>

And directly convert it into lines that can be plotted (the dimensions for this are for the [Line-us drawing arm](https://github.com/Line-us/Line-us-Programming/blob/master/Documentation/LineUsDrawingArea.pdf)):

![](.github/output.gif)

## Install

You need the following requirements:

- python (3.6+)
- potrace
- ffmpeg
- imagemagick

On Windows you can install with [scoop](https://scoop.sh/). Note that you must use `python37` to install Python because one of the libraries (simplification) on Windows is incompatible with Python3.8.

	scoop install potrace ffmpeg imagemagick python37 make

On Linux you can install with apt:
	
	sudo apt install potrace ffmpeg imagemagick python3 make

Once installed, you can install the required Python packages with pip:

	python3 -m pip install click loguru matplotlib numpy simplification svgpathtools svgwrite tqdm

## Run

You can just run:

	python3 run.py --file image.jpg --animate

After it runs it should create a folder `image.jpg.drawbot`. In that folder you should have `movie.mp4` which is a movie showing how the line drawing will be made,  `procesesd.svg` which shows the processed svg with coordinates mapped to the drawing area.


## License

MIT