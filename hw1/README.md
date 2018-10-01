# Computer Vision

Lecturer: Prof. [Shao-Yi Chien](http://media.ee.ntu.edu.tw/member.html)<br>
Course website: http://media.ee.ntu.edu.tw/courses/cv/18F/.html

## Tasks
* Task1: conventional rgb2gray, please check the argument listed in `rgb2gray.py`
* Task2: advanced rgb2gray, please check the argument listed in `adv_rgb2gray.py`

## Rerun the code
		python3 rgb2gray.py --source_path [the path of rgb image folder] --target_path [the folder path to save grayscale images] --img_name [the filename of .png image] 
		python3 adv_rgb2gray.py --source_path [the path of rgb image folder] --target_path [the folder path to save grayscale images] --img_name [the filename of .png image] --already_done [0 for rerunning and 1 for testing]


## Result
* Task1 <br>
rgb image:
<img src="https://github.com/JerryHoTaiwan/Computer-Vision/blob/master/hw1/testdata/0b.png" width="250">
grayscale image:
<img src="https://github.com/JerryHoTaiwan/Computer-Vision/blob/master/hw1/result/0b_y.png" width="250">

* Task2 <br>
src image:
<img src="https://github.com/JerryHoTaiwan/Computer-Vision/blob/master/hw1/testdata/0a.png" width="250">
candidate image:
<img src="https://github.com/JerryHoTaiwan/Computer-Vision/blob/master/hw1/result/0a/can/0a_y0.png" width="250">
guided image:
<img src="https://github.com/JerryHoTaiwan/Computer-Vision/blob/master/hw1/result/0a/res/0a_y0.png" width="250">
the cost on the surface of weight space:
<img src="https://github.com/JerryHoTaiwan/Computer-Vision/blob/master/hw1/result/0a/surface/sf_0.png" width="250">

## Reference
Decolorization: Is rgb2gray() Out?[[link]](https://ybsong00.github.io/siga13tb/siga13tb_final.pdf)<br>
