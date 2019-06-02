import os
images = os.listdir('progress')
from PIL import Image, ImageDraw, ImageFont
(x, y) = (185, 10)
color = 'rgb(0, 0, 0)' 
for img in images:
    image = Image.open('progress/'+img)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Roboto-Regular.ttf', size=15)
    no = img.split("_")[1].split('.')[0]
    message = "EPOCH {}".format(no)
    draw.text((x, y), message, fill=color, font=font)
    image.save('progress/'+img)

#convert to gif online