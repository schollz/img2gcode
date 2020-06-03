

generate:
	convert 1.jpg -resize 2000x1125 -background White -gravity center -extent 2000x1125 -threshold 60% out.bmp
	convert out.bmp -negate -morphology Thinning:-1 Skeleton out2.bmp
	convert out2.bmp -negate out3.bmp
	convert out3.bmp -shave 1x1 -bordercolor black -border 1 -rotate 90 out4.bmp
	potrace -b svg -o out.svg out4.bmp
	python3 run.py --svgin out.svg --svgout output2.svg --animate 1.mp4
	ffmpeg -i 1.mp4 -vf "transpose=2" -y 2.mp4

gifit:
	ffmpeg -i 2.mp4 -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 1 -y output.gif
	ffmpeg -i output.gif -y 4.mp4
	ffmpeg -i 4.mp4 -filter:v "crop=800:500:0:300" -y 5.mp4
	ffmpeg -i 5.mp4 -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 1 -y output2.gif
	rm 4.mp4 5.mp4

