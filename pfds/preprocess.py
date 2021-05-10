import os
from os import listdir
import pydicom
import numpy as np
from skimage import measure
from skimage.morphology import disk, opening
from PIL import Image

import tensorflow as tf
import pydicom
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate 
)
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

import efficientnet.tfkeras as efn
config = tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)

############# PRE-PROCESS IMAGE (RESIZE) ##########################
class Pre_image():
    def __init__(self):
        #self.image_path = image_path
        self.my_shape = (64,64)
    def load_scans(self, image_path):
        aslice = pydicom.dcmread(image_path) 
        return aslice
    def set_outside_scanner_to_air(self,raw_pixelarrays):
        raw_pixelarrays[raw_pixelarrays<=-1000] = 0
        return raw_pixelarrays
    def transform_to_hu(self, slices):
        images = np.stack([file.pixel_array for file in slices])
        images= images.astype(np.int16)
        images = self.set_outside_scanner_to_air(images)
        #convert to HU
        for n in range(len(slices)):
            intercept = slices[n].RescaleIntercept
            slope = slices[n].RescaleSlope

            if slope !=1:
                images[n] = slope*images[n].astype(np.float64)
                images[n] = images.astype(np.int16)
        
        images[n] += np.int16(intercept)
        return np.array(images, dtype=np.int16)
    def get_window_value(self,feature):
        if type(feature) == pydicom.multival.MultiValue:
            return np.int(feature[0])
        else:
            return np.int(feature)
    def resize_scan(self,scan):
        # read slice as 32 bit signed integers
        img = Image.fromarray(scan, mode = "I")
        # do the resizing
        img = img.resize(self.my_shape, resample=Image.LANCZOS)
        # convert back to 16 bit integers
        resized_scan = np.array(img, dtype=np.int16)
        return resized_scan
    def crop_scan(self,scan):
        img = Image.fromarray(scan, mode="I")
        left = (scan.shape[0]-512)/2
        right = (scan.shape[0]+512)/2
        top = (scan.shape[1]-512)/2
        bottom = (scan.shape[1]+512)/2

        img = img.crop((left, top, right, bottom))
        # convert back to 16 bit integers
        cropped_scan = np.array(img, dtype=np.int16)
        return cropped_scan
    def crop_and_resize(self, scan):
        img = Image.fromarray(scan, mode="I")
        
        left = (scan.shape[0]-512)/2
        right = (scan.shape[0]+512)/2
        top = (scan.shape[1]-512)/2
        bottom = (scan.shape[1]+512)/2
    
        img = img.crop((left, top, right, bottom))
        img = img.resize(self.my_shape, resample=Image.LANCZOS)
        
        cropped_resized_scan = np.array(img, dtype=np.int16)
        return cropped_resized_scan   
    def segment_lung_mask(self,image):
        segmented = np.zeros(image.shape)

        for n in range(image.shape[0]):
            binary_image = np.array(image[n]>-320, dtype= np.int8) +1
            labels = measure.label(binary_image)
            bad_labels = np.unique([labels[0,:], labels[-1,:], labels[:,0],labels[:,-1]])
            for bad_label in bad_labels:
                binary_image[labels == bad_label ] = 2
                selem = disk(2)
                binary_image = opening(binary_image, selem)

                binary_image -=1
                binary_image = 1- binary_image
                segmented[n] = binary_image.copy()* image[n]

            return segmented  
class IGenerator(Sequence):
    def __init__(self, keys, a, tab, batch_size=32):
        self.keys = [k for k in keys]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        x = []
        a,tab = [],[]
        keys = np.random.choice(self.keys, size = self.batch_size)
        for k in keys:
            img = segmented_array
            i1 = np.random.randint(int(img.shape[0]))
            x.append(img[i1])
            a.append(self.a[k])
            tab.append(self.tab[k])
        x,a,tab = np.array(x), np.array(a),np.array(tab)
        x= np.expand_dims(x, axis=-1)
        print(x.shape)
        return [x, tab], a
    def get_tab(ppatient):
        vector = [(ppatient.ages - 30) / 30] 
        if ppatient.gender == 'male':
            vector.append(0)
        else:
            vector.append(1)
        if ppatient.smoke == 'Never smoked':
            vector.extend([0,0])
        elif ppatient.smoke == 'Ex-smoker':
            vector.extend([1,1])
        elif ppatient.smoke == 'Currently smokes':
            vector.extend([0,1])
        else:
            vector.extend([1,0])
        return np.array(vector)

###########     LOAD EFFICIENT NET MODEL      ###########
class Load_Eff():
    def __init__(self,shape, weights_file):
        self.shape = shape
        self.model_class ='b2'
        self.weights = weights_file
    def get_efficientnet(self):
        models_dict ={
            'b0': efn.EfficientNetB0(input_shape=self.shape,weights=None,include_top=False),
            'b1': efn.EfficientNetB1(input_shape=self.shape,weights=None,include_top=False),
            'b2': efn.EfficientNetB2(input_shape=self.shape,weights=None,include_top=False),
            'b3': efn.EfficientNetB3(input_shape=self.shape,weights=None,include_top=False),
            'b4': efn.EfficientNetB4(input_shape=self.shape,weights=None,include_top=False),
            'b5': efn.EfficientNetB5(input_shape=self.shape,weights=None,include_top=False),
            'b6': efn.EfficientNetB6(input_shape=self.shape,weights=None,include_top=False),
            'b7': efn.EfficientNetB7(input_shape=self.shape,weights=None,include_top=False)
        }
        return models_dict[self.model_class]
    def build_model(self):
        inp = Input(shape=self.shape)
        base = self.get_efficientnet()
        x = base(inp)
        x = GlobalAveragePooling2D()(x)
        inp2 = Input(shape=(4,))
        x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
        x = Concatenate()([x, x2]) 
        x = Dropout(0.5)(x) 
        x = Dense(1)(x)
        model = Model([inp, inp2] , x)
        model.load_weights(self.weights)
        return model
#########LOAD QUANTIAL NET MODEL###################
class Load_Qt():
    def __init__(self, weights_file):
        self.weights_file = weights_file
        self.C1 = tf.constant(70, dtype='float32')
        self.C2 =  tf.constant(1000, dtype="float32") 

    def score(self,y_true,y_pred):
        tf.dtypes.cast(y_true, tf.float32)
        tf.dtypes.cast(y_pred, tf.float32)
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]
        sigma_clip = tf.maximum(sigma, self.C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, self.C2)
        sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
        metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
        return K.mean(metric)
    def qloss(self,y_true, y_pred):
        # Pinball loss for multiple quantiles
        qs = [0.2, 0.50, 0.8]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q*e, (q-1)*e)
        return K.mean(v)
    def mloss(self,_lambda):
        def loss(y_true, y_pred):
            return _lambda* self.qloss(y_true,y_pred) +(1-_lambda)* self.score(y_true,y_pred)
        return loss
    def make_model(self,nh):
        z = L.Input((nh,), name = "Patient")
        x = L.Dense(100, activation= "relu", name = "d1")(z)
        x = L.Dense(100,activation = "relu", name = "d2")(x)
        p1 = L.Dense(3, activation="linear", name="p1")(x)
        p2 = L.Dense(3, activation="relu", name="p2")(x)
        preds =  L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
        model = M.Model(z, preds, name = "CNN" )
        model.compile(loss = self.mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[])
        return model


'''
        detect = AbnDetect()
        detect.image_path = os.path.join(settings.MEDIA_ROOT,abnpatient.xray.path)
        detect.model_path =  os.path.join(settings.MODEL_ROOT,'csv_retinanet_106.pt')
        detect.class_list = os.path.join(settings.MODEL_ROOT, 'class_list.csv')
        img = detect.detect_image()
'''