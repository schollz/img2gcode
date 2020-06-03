

generate:
	convert 1.jpg -resize 2000x1125 -background White -gravity center -extent 2000x1125 -threshold 60% out.bmp
	convert out.bmp -negate -morphology Thinning:-1 Skeleton out2.bmp
	convert out2.bmp -negate out3.bmp
	convert out3.bmp -shave 1x1 -bordercolor black -border 1 -rotate 90 out4.bmp
	potrace -b svg -o out.svg out4.bmp
	python3 run.py
	ffmpeg -i line.mp4 -vf "transpose=2" -y line2.mp4