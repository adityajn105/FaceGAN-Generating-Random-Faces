from flask import Flask, render_template, request, send_from_directory
from build_generator import Generator
import os
import matplotlib.pyplot as plt
import numpy as np
import base64

app = Flask(__name__, template_folder=os.curdir)
generator=None


@app.route('/')
def home():
    noise = np.random.normal(np.random.normal(0,1,(1,100)))
    img = (generator.predict(noise)[0]/2)+0.5
    img = np.clip(img,0,1)
    plt.imsave('temp/temp.png',img)
    encoded_string=''
    with open("temp/temp.png", "rb") as image_file:
        encoded_string += str(base64.b64encode(image_file.read()))[2:-1]
    return render_template('facegen.html',data = {'img':encoded_string})

@app.route('/<path:path>')
def getStaticFiles(path):
    print(path)
    return send_from_directory(os.curdir, path)

if __name__ == "__main__":
    generator = Generator('../saved_weights/generator_weights.h5')
    print('Generator Built!! You are ready with app.')
    app.run(host = '0.0.0.0',port = int(5000))