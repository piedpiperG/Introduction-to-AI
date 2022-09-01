from PIL import Image
#图像裁剪
im=Image.open("d:\\test.jpg")
im_L = im.convert("L")
box = (560,1000,1800,1800)
region = im_L.crop(box)
region.save("d:\\crop_img.jpg")
region.show()
#图像合并，此处 1.jpg 与 2.jpg 所有通道必须有相同的尺寸
im1 = Image.open("d:\\1.jpg")
im2 = Image.open("d:\\1.jpg")
r1,g1,b1 = im1.split()
r2,g2,b2 = im2.split()
print(r1.mode,r1.size,g1.mode,g1.size)
print(r2.mode,r2.size,g2.mode,g2.size)
new_im=[r1,g2,b2]
print(len(new_im))
im_merge = Image.merge("RGB",new_im)
im_merge.show()
from PIL import Image
from pylab import *
# 读取图像到数组中
im = array(Image.open('d:\\test.jpg'))
# 绘制图像
imshow(im)
# 一些点
x = [100,100,400,400]
y = [200,500,200,500]
# 使用红色星状标记绘制点
plot(x,y,'r*')
# 绘制连接前两个点的线
plot(x[:2],y[:2])
# 添加标题，显示绘制的图像
title('Plotting: "empire.jpg"')
show()
from PIL import Image
from PIL import ImageEnhance
#原始图像
image = Image.open("d:\\1.jpg")
image.show()
#亮度增强
enh_bri = ImageEnhance.Brightness(image)
brightness = 1.5
image_brightened = enh_bri.enhance(brightness)
image_brightened.show()
#对比度增强
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
#原始图像
image = Image.open("d:\\1.jpg")
image.show()
#对比度增强
enh_con = ImageEnhance.Contrast(image)
contrast = 1.5
image_contrasted = enh_con.enhance(contrast)
image_contrasted.show()
#锐度增强
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 3.0
image_sharped = enh_sha.enhance(sharpness)
image_sharped.show()
#图像模糊
im = Image.open("d:\\1.jpg")
im_blur = im.filter(ImageFilter.BLUR)
im_blur.show()
#轮廓提取
im = Image.open("d:\\1.jpg")
im_contour = im.filter(ImageFilter.CONTOUR)
im_contour.show()
#画直方图
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img=np.array(Image.open('d:/test.jpg').convert('L'))
plt.figure("lena")
arr=img.flatten()
n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green',
alpha=0.75)
plt.show()
# 图像分割
# 图片二值化
from PIL import Image
img = Image.open('d:\\test.jpg')
# 模式 L”为灰色图像，它的每个像素用 8 个 bit 表示，0 表示黑，255 表示白，其他数字表
示不同的灰度。
Img = img.convert('L')
Img.save("d:\\test1.jpg")
# 自定义灰度界限，大于这个值为黑色，小于这个值为白色
threshold = 200
table = []
for i in range(256):
 if i < threshold:
    table.append(0)
 else:
    table.append(1)
# 图片二值化
photo = Img.point(table, '1')
photo.show()
#验证码识别
import pytesseract
from PIL import Image
#1.引入 Tesseract 程序
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files
(x86)\Tesseract-OCR\tesseract.exe'
#2.使用 Image 模块下的 Open()函数打开图片
image = Image.open('d:\\6.jpg',mode='r')
print(image)
#3.识别图片文字
code= pytesseract.image_to_string(image)
print(code)