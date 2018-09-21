#!/home/eindemwort/anaconda3/bin/python3

import web
import os
import keras

from web import form
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np

urls = ('/', 'DogApp',
        '/pred', 'Prediction')
render = web.template.render('templates/',)

register_form = form.Form(
    form.Button("make another prediction!", type="submit", description="Register")
)


class DogApp:

    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return render.upload('')


    def POST(self):
        x = web.input(myfile={})
        filedir = './static' # change this to the directory you want to store the file in.
        if 'myfile' in x: # to check if the file-object is created
            filepath=x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
            #filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
            filename="theImage"
            fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
            fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.

            infile = filedir +'/'+filename
            outfile = filedir +'/'+filename
            im = Image.open(filedir +'/'+filename)
            im.thumbnail((280, 280))
            im.save(outfile, im.format)

        #return render.upload(outfile)
        raise web.seeother('/pred')


class Prediction:
    def GET(self):
        web.header("Content-Type", "text/html; charset=utf-8")
        filedir = "./static"
        filename="theImage"
        outfile = filedir+'/'+filename

        f = register_form()
        return render.prediction(f,outfile)


    def POST(self):
        filedir = "./static"
        filename="theImage"
        outfile = filedir+'/'+filename

        f = register_form()
        os.remove(outfile)
        raise web.seeother('/')


if __name__ == "__main__":
   app = web.application(urls, globals(), True)
   app.run()
