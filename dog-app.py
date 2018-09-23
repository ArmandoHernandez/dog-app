#!/home/eindemwort/anaconda3/bin/python3

import web
import os
import keras

from web import form
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input
from multiprocessing.managers import BaseManager, NamespaceProxy

import numpy as np
import tensorflow as tf

urls = ('/', 'DogApp',
        '/pred', 'Prediction')
render = web.template.render('templates/',)

register_form = form.Form(
    form.Button("make another prediction!", type="submit", description="Register")
)


class Utilities:
    @staticmethod
    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    @staticmethod
    def getBottlenecks(tensor):
        from keras.applications.xception import Xception, preprocess_input
        return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


class Models:
    def __init__(self):
        self.resnet50_model = ResNet50()
        self.resnet50_model.load_weights('static/saved_models/ResNet50_model_weights.h5')

        self.human_detector = Sequential()
        self.human_detector.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3)))
        self.human_detector.add(MaxPooling2D(pool_size=2))
        self.human_detector.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        self.human_detector.add(MaxPooling2D(pool_size=2))
        self.human_detector.add(Dropout(0.1))
        self.human_detector.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        self.human_detector.add(MaxPooling2D(pool_size=2))
        self.human_detector.add(Dropout(0.3))
        self.human_detector.add(Flatten())
        self.human_detector.add(Dense(64, activation='relu'))
        self.human_detector.add(Dropout(0.2))
        self.human_detector.add(Dense(1, activation='sigmoid'))
        self.human_detector.compile(loss='binary_crossentropy', optimizer='adamax',
             metrics=['accuracy'])
        self.human_detector.load_weights("static/saved_models/human_detector_weights.h5")

        self.Xception_model = Sequential()
        self.Xception_model.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
        self.Xception_model.add(Dense(133, activation='softmax'))
        self.Xception_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        self.Xception_model.load_weights('static/saved_models/weights.best.Xception.hdf5')

        self.graph = tf.get_default_graph()

    def res_net_predict(self,img):
        with self.graph.as_default():
            prediction = self.resnet50_model.predict(img)
        return prediction

    def human_detector_predict(self,img_path):
        with self.graph.as_default():
            prediction = (self.human_detector.predict(Utilities.path_to_tensor(img_path)) > 0.60)
        return prediction

    def breed_detector_predict(self,img_path):
        with self.graph.as_default():
            predicted_vector = self.Xception_model.predict(Utilities.getBottlenecks(Utilities.path_to_tensor(img_path)))
        breeds_list = open("static/breeds_list/dog_breed_list.txt").readlines()
        str = breeds_list[np.argmax(predicted_vector)]
        return str[str.rfind(".") + 1:]


class MyManager(BaseManager): pass
MyManager.register('resnet', Models, exposed=('res_net_predict','human_detector_predict', 'breed_detector_predict',))


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
            im.thumbnail((400, 400))
            im.save(outfile, im.format)

        #return render.upload(outfile)
        raise web.seeother('/pred')


class Prediction:
    def GET(self):
        web.header("Content-Type", "text/html; charset=utf-8")
        manager = MyManager()
        manager.start()
        resnet_manager = manager.resnet()

        filedir = "./static"
        filename="theImage"
        outfile = filedir+'/'+filename
        is_dog = False
        is_human = False
        breed = None
        perform_prediction = True

        img = preprocess_input(Utilities.path_to_tensor(outfile))

        class_predicted = np.argmax(resnet_manager.res_net_predict(img))
        is_dog = ((class_predicted <= 268) & (class_predicted >= 151))

        if is_dog == False:
            is_human = resnet_manager.human_detector_predict(outfile)
            if is_human == False:
                perform_prediction = False

        if perform_prediction == True:
            breed = resnet_manager.breed_detector_predict(outfile)

        f = register_form()

        return render.prediction(f,outfile, is_dog, is_human, breed)


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
