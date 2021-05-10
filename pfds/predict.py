
import os
import tensorflow as tf
import pydicom
from pydicom import dcmread
import numpy as np 
import pandas as pd 
from skimage import measure 
from skimage.morphology import disk, opening, closing,dilation,erosion
from sklearn.metrics import mean_absolute_error
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from os import listdir, mkdir
#from tqdm.notebook import tqdm
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
config.gpu_options.allow_growth = False
session = tf.compat.v1.Session(config=config)

################################################################################
##############                 FILE PATH                        ################
################################################################################

WEIGHT_EFFICIENT_NET = 'C:/Users/Carol/Desktop/DEMO_FIBRO/effnet_30.h5'
MODEL_QUANTIAL = "C:/Users/Carol/Desktop/DEMO_FIBRO/savelastmodel.h5"

PATIENT_DATA = "C:/Users/Carol/Desktop/DEMO_FIBRO/demo.csv"
SUBMISSION = "C:/Users/Carol/Desktop/DEMO_FIBRO/submission.csv"
EFFICIENT_SUBMISSION = "C:/Users/Carol/Desktop/DEMO_FIBRO/efficient_sub.csv"
QUANTIAL_SUBMISSION = "C:/Users/Carol/Desktop/DEMO_FIBRO/quantile_submission.csv"
FINAL_SUBMISSION = 'C:/Users/Carol/Desktop/DEMO_FIBRO/FINAL_PREDICT.csv'

PATIENT_CT_SCANS_FOLDER = "C:/Users/Carol/Desktop/DEMO_FIBRO/demo/"

TRAINING_CSV = "C:/Users/Carol/Desktop/DEMO_FIBRO/train.csv" #JUST USE FOR GETTING ALL OF VALUE OF [SEX, SMOKING STATUS]

################################################################################
###############           PRE-PROCESS IMAGE (RESIZE)            ################
################################################################################
class DFDetect():
    def __init__(self,weights_eff,weights_quantial, pdata,pct_forlder):
        self.weights_eff = weights_eff
        self.weights_quantial = weights_quantial
        self.pdata = pdata
        self.pct_forlder = pct_forlder
    def load_scans(self,dcm_path):
    # in this competition we have missing values in ImagePosition, this is why we are sorting by filename number
        files = listdir(dcm_path)
        file_nums = [np.int(file.split(".")[0]) for file in files]
        sorted_file_nums = np.sort(file_nums)[::-1]
        slices = [pydicom.dcmread(dcm_path + "/" + str(file_num) + ".dcm" ) for file_num in sorted_file_nums]
        return slices
    #Convert từ pixel sang HU
    def set_outside_scanner_to_air(self,raw_pixelarrays):
        # in OSIC we find outside-scanner-regions with raw-values of -2000. 
        # Let's threshold between air (0) and this default (-2000) using -1000
        raw_pixelarrays[raw_pixelarrays <= -1000] = 0
        return raw_pixelarrays
    def transform_to_hu(self,slices):
        images = np.stack([file.pixel_array for file in slices])
        images = images.astype(np.int16)
        images = self.set_outside_scanner_to_air(images)
        
        # convert to HU
        for n in range(len(slices)):
            
            intercept = slices[n].RescaleIntercept
            slope = slices[n].RescaleSlope
            
            if slope != 1:
                images[n] = slope * images[n].astype(np.float64)
                images[n] = images[n].astype(np.int16)
            
            images[n] += np.int16(intercept)
            
        return np.array(images, dtype=np.int16)
    def get_window_value(self,feature):
        if type(feature) == pydicom.multival.MultiValue:
            return np.int(feature[0])
        else:
            return np.int(feature)

    def resize_scan(scan, new_shape):
        # read slice as 32 bit signed integers
        img = Image.fromarray(scan, mode="I")
        # do the resizing
        img = img.resize(new_shape, resample=Image.LANCZOS)
        # convert back to 16 bit integers
        resized_scan = np.array(img, dtype=np.int16)
        return resized_scan
    def crop_scan(scan):
        img = Image.fromarray(scan, mode="I")
        left = (scan.shape[0]-512)/2
        right = (scan.shape[0]+512)/2
        top = (scan.shape[1]-512)/2
        bottom = (scan.shape[1]+512)/2

        img = img.crop((left, top, right, bottom))
        # convert back to 16 bit integers
        cropped_scan = np.array(img, dtype=np.int16)
        return cropped_scan
    def crop_and_resize(scan, new_shape):
        img = Image.fromarray(scan, mode="I")
        
        left = (scan.shape[0]-512)/2
        right = (scan.shape[0]+512)/2
        top = (scan.shape[1]-512)/2
        bottom = (scan.shape[1]+512)/2
        
        img = img.crop((left, top, right, bottom))
        img = img.resize(new_shape, resample=Image.LANCZOS)
        
        cropped_resized_scan = np.array(img, dtype=np.int16)
        return cropped_resized_scan
    def preprocess_to_hu_scans(scan_properties, my_shape):
        
        patient_pth=[]
        for patient in enumerate(scan_properties.patient.values):
            patient_pth.append(demo[demo.Patient == patient].dcm_path.values)
        for i, patient in enumerate(scan_properties.patient.values):
            pth = scan_properties.loc[scan_properties.patient==patient].patient_pth.values[0]
            scans = load_scans(pth)
            hu_scans = transform_to_hu(scans) 
            prepared_scans = np.zeros((hu_scans.shape[0], my_shape[0], my_shape[1]), dtype=np.int16)
            hu_scans = hu_scans.astype(np.int32)
            for s in range(hu_scans.shape[0]):
            prepared_scans[s]=resize_scan(hu_scans[s,:,:],my_shape)
            return prepared_scans
            
            # if squared:
            if hu_scans.shape[1] == hu_scans.shape[2]:
                # if size is as desired
                if hu_scans.shape[1] == my_shape[0]:
                    continue
                # else resize:
                else:
                # as we have not converted to jpeg to keep all information, we need to do a workaround
                    hu_scans = hu_scans.astype(np.int32)
                    for s in range(hu_scans.shape[0]): 
                        prepared_scans[s] = resize_scan(hu_scans[s,:,:], my_shape)

            # if non-squared - do a center crop to 512, 512 and then resize to desired shape
            else:
                hu_scans = hu_scans.astype(np.int32)
                for s in range(hu_scans.shape[0]):
                    # if desired shape is 512x512:
                    if my_shape[0]==512:
                        prepared_scans[s] = crop_scan(hu_scans[s,:,:])
                    else:
                        prepared_scans[s] = crop_and_resize(hu_scans[s,:,:], my_shape)
                    
            # save the prepared scans of patient:
            return prepared_scans

    demo = pd.read_csv(PATIENT_DATA)
    demo["dcm_path"] = PATIENT_CT_SCANS_FOLDER + demo.Patient + "/"

    N=demo.shape[0]

    pixelspacing_r = []
    pixelspacing_c = []
    slice_thicknesses = [] #List chứa độ dày lớp cắt của ảnh chụp CT từng bệnh nhân
    patient_id = []
    patient_pth = []
    row_values = []
    column_values = []
    window_widths = []
    window_levels = []

    patients = demo.Patient.unique()[0:N]

    for patient in patients:
        patient_id.append(patient)
        path = demo[demo.Patient == patient].dcm_path.values[0]
        example_dcm = listdir(path)[0]  #Ảnh dcm ví dụ
        patient_pth.append(path) #Đường dẫn tới thư mục CT của bệnh nhân
        dataset = pydicom.dcmread(path + "/" + example_dcm) #dataset về ảnh CT của bệnh nhân
        
        window_widths.append(get_window_value(dataset.WindowWidth)) 
        window_levels.append(get_window_value(dataset.WindowCenter))
        
        spacing = dataset.PixelSpacing
        slice_thicknesses.append(dataset.SliceThickness)
        
        row_values.append(dataset.Rows)
        column_values.append(dataset.Columns)
        pixelspacing_r.append(spacing[0])
        pixelspacing_c.append(spacing[1])
        
    scan_properties = pd.DataFrame(data=patient_id, columns=["patient"])
    # print(scan_properties)
    scan_properties.loc[:, "rows"] = row_values
    scan_properties.loc[:, "columns"] = column_values
    scan_properties.loc[:, "area"] = scan_properties["rows"] * scan_properties["columns"]
    scan_properties.loc[:, "pixelspacing_r"] = pixelspacing_r
    scan_properties.loc[:, "pixelspacing_c"] = pixelspacing_c
    scan_properties.loc[:, "pixelspacing_area"] = scan_properties.pixelspacing_r * scan_properties.pixelspacing_c
    scan_properties.loc[:, "slice_thickness"] = slice_thicknesses
    scan_properties.loc[:, "patient_pth"] = patient_pth
    scan_properties.loc[:, "window_width"] = window_widths
    scan_properties.loc[:, "window_level"] = window_levels

    img_resized = preprocess_to_hu_scans(scan_properties, (64,64))

    #########################################################################
    ##########         PRE-PROCESS IMAGE (LUNG SEGMENTATION)        #########
    #########################################################################

    def segment_lung_mask(image):
        segmented = np.zeros(image.shape)   
        
        for n in range(image.shape[0]):
            binary_image = np.array(image[n] > -320, dtype=np.int8)+1
            labels = measure.label(binary_image)
            
            bad_labels = np.unique([labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])
            for bad_label in bad_labels:
                binary_image[labels == bad_label] = 2
        
            #We have a lot of remaining small signals outside of the lungs that need to be removed. 
            #In our competition closing is superior to fill_lungs 
            selem = disk(2)
            binary_image = opening(binary_image, selem)
        
            binary_image -= 1 #Make the image actual binary
            binary_image = 1-binary_image # Invert it, lungs are now 1

            
            segmented[n] = binary_image.copy() * image[n]
        
        return segmented


    segmented_array=segment_lung_mask(img_resized)

    #######################################################################
    #############       LOAD EFFICIENT NET MODEL                ###########
    #######################################################################
    class IGenerator(Sequence):
        BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
        def __init__(self, keys, a, tab, batch_size=32):
            self.keys = [k for k in keys if k not in self.BAD_ID]
            self.a = a
            self.tab = tab
            self.batch_size = batch_size
            
        
        def __len__(self):
            return 1000
        
        def __getitem__(self, idx):
            x = []
            a, tab = [], [] 
            keys = np.random.choice(self.keys, size = self.batch_size)
            for k in keys:
                img = segmented_array
                i1=np.random.randint(int(img.shape[0]))
                x.append(img[i1])
                # for i in range(2):
                a.append(self.a[k])
                tab.append(self.tab[k])

        
            x,a,tab = np.array(x), np.array(a), np.array(tab)
            x = np.expand_dims(x, axis=-1)
            print(x.shape)
            return [x, tab] , a

    def get_tab(df):
        vector = [(df.Age.values[0] - 30) / 30] 
        
        if df.Sex.values[0] == 'male':
        vector.append(0)
        else:
        vector.append(1)
        
        if df.SmokingStatus.values[0] == 'Never smoked':
            vector.extend([0,0])
        elif df.SmokingStatus.values[0] == 'Ex-smoker':
            vector.extend([1,1])
        elif df.SmokingStatus.values[0] == 'Currently smokes':
            vector.extend([0,1])
        else:
            vector.extend([1,0])
        return np.array(vector)

    def get_efficientnet(model, shape):
        models_dict = {
            'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
            'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
            'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
            'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
            'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
            'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
            'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
            'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)
        }
        return models_dict[model]

    def build_model(shape=(64, 64, 1), model_class=None):
        inp = Input(shape=shape)
        base = get_efficientnet(model_class, shape)
        x = base(inp)
        x = GlobalAveragePooling2D()(x)
        inp2 = Input(shape=(4,))
        x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
        x = Concatenate()([x, x2]) 
        x = Dropout(0.5)(x) 
        x = Dense(1)(x)
        model = Model([inp, inp2] , x)
        model.load_weights(WEIGHT_EFFICIENT_NET)
        return model


    A = {} 
    TAB = {} 
    P = [] 
    for i, p in enumerate(demo.Patient.unique()):
        sub = demo.loc[demo.Patient == p, :] 
        fvc = sub.FVC.values
        weeks = sub.Weeks.values
        #y=mx+c => fvc= week*a +b
        c = np.vstack([weeks, np.ones(len(weeks))]).T
        a, b = np.linalg.lstsq(c, fvc)[0]
        
        A[p] = a
        TAB[p] = get_tab(sub)
        P.append(p)

    model_classes = ['b2'] #['b0','b1','b2','b3',b4','b5','b6','b7']
    models = [build_model(shape=(64, 64, 1), model_class=m) for m in model_classes]
    subs = []
    for model in models:

        q = 0.5
        sub = pd.read_csv(SUBMISSION)
        A_test, B_test, P_test,W, FVC= {}, {}, {},{},{} 
        STD, WEEK = {}, {} 
        for p in demo.Patient.unique():
            x = [] 
            tab = [] 
            img_one = segmented_array
            for dim in range(img_one.shape[0]):
                    x.append(img_one[dim]) 
                    tab.append(get_tab(demo.loc[demo.Patient == p, :])) 
            if len(x) <= 1:
                continue
            tab = np.array(tab) 

            x = np.expand_dims(x, axis=-1) 
            _a = model.predict([x, tab]) 
            # print(_a)
            a = np.quantile(_a, q)
            # print("a=",a)
            A_test[p] = a
            B_test[p] = demo.FVC.values[demo.Patient == p] - a*demo.Weeks.values[demo.Patient == p]
            P_test[p] = demo.Percent.values[demo.Patient == p] 
            WEEK[p] = demo.Weeks.values[demo.Patient == p]

        for k in sub.Patient_Week.values:
            p, w = k.split('_') #pは患者で、ｗは週間です。
            w = int(w) 

            fvc = A_test[p] * w + B_test[p]
            sub.loc[sub.Patient_Week == k, 'FVC'] = fvc
            sub.loc[sub.Patient_Week == k, 'Confidence'] = (
                P_test[p] - A_test[p] * abs(WEEK[p] - w) 
        )   
            # if(p == 'ID00421637202311550012437'):
            #   print(fvc)

        _sub = sub[["Patient_Week","FVC","Confidence"]].copy()
        subs.append(_sub)

    ###############################################################################
    ############                SAVE EFFICIENT NET PREDICT          ###############
    ###############################################################################
    N = len(subs)
    sub = subs[0].copy() # ref
    sub["FVC"] = 0
    sub["Confidence"] = 0
    for i in range(N):
        sub["FVC"] += subs[0]["FVC"] * (1/N)
        sub["Confidence"] += subs[0]["Confidence"] * (1/N)

    sub[["Patient_Week","FVC","Confidence"]].to_csv(EFFICIENT_SUBMISSION, index=False)

    img_sub = sub[["Patient_Week","FVC","Confidence"]].copy()

    #################################################################################
    #########               LOAD QUANTIAL NET MODEL              ####################
    #################################################################################

    C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

    def score(y_true, y_pred):
        tf.dtypes.cast(y_true, tf.float32)
        tf.dtypes.cast(y_pred, tf.float32)
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]
        
        #sigma_clip = sigma + C1
        sigma_clip = tf.maximum(sigma, C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, C2)
        sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
        metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
        return K.mean(metric)

    def qloss(y_true, y_pred):
        # Pinball loss for multiple quantiles
        qs = [0.2, 0.50, 0.8]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q*e, (q-1)*e)
        return K.mean(v)

    def mloss(_lambda):
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
        return loss

    def make_model(nh):
        z = L.Input((nh,), name="Patient")
        x = L.Dense(100, activation="relu", name="d1")(z)
        x = L.Dense(100, activation="relu", name="d2")(x)
        #x = L.Dense(100, activation="relu", name="d3")(x)
        p1 = L.Dense(3, activation="linear", name="p1")(x)
        p2 = L.Dense(3, activation="relu", name="p2")(x)
        preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                        name="preds")([p1, p2])
        
        model = M.Model(z, preds, name="CNN")
        #model.compile(loss=qloss, optimizer="adam", metrics=[score])
        model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
        return model

    # BATCH_SIZE=128

    tr = pd.read_csv(TRAINING_CSV)
    tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
    chunk = pd.read_csv(PATIENT_DATA)

    print("add infos")
    sub = pd.read_csv(EFFICIENT_SUBMISSION)
    sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
    sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
    sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
    sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")

    tr['WHERE'] = 'train'
    chunk['WHERE'] = 'val'
    sub['WHERE'] = 'test'
    data = chunk.append([sub,tr])

    data['min_week'] = data['Weeks']
    data.loc[data.WHERE=='test','min_week'] = np.nan
    data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

    base = data.loc[data.Weeks == data.min_week]
    base = base[['Patient','FVC']].copy()
    base.columns = ['Patient','min_FVC']
    base['nb'] = 1
    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
    base = base[base.nb==1]
    base.drop('nb', axis=1, inplace=True)

    data = data.merge(base, on='Patient', how='left')
    data['base_week'] = data['Weeks'] - data['min_week']
    del base

    COLS = ['Sex','SmokingStatus'] #,'Age'
    FE = []
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(float)

    data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
    data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )
    data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
    #data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )
    FE += ['age','week','BASE']

    tr = data.loc[data.WHERE=='train']
    chunk = data.loc[data.WHERE=='val']
    sub = data.loc[data.WHERE=='test']
    del data


    y = tr['FVC'].values
    z = tr[FE].values
    ze = sub[FE].values
    nh = ze.shape[1]
    pe = np.zeros((ze.shape[0], 3))
    pred = np.zeros((z.shape[0], 3))

    net = make_model(nh)
    net.load_weights(MODEL_QUANTIAL)
    pe = net.predict(ze, batch_size=128, verbose=0) #/ 2 #NFOLD in training process is 2
    pred = net.predict(z, batch_size=128, verbose=0)

    ###################################################################################
    #################               SAVE QUANTIAL PREDICT               ###############
    ###################################################################################

    sigma_opt = mean_absolute_error(y, pred[:, 1])
    unc = pred[:,2] - pred[:, 0]
    sigma_mean = np.mean(unc)

    sub['FVC1'] = 1.*pe[:, 1]
    sub['Confidence1'] = pe[:, 2] - pe[:, 0]
    subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
    subm.loc[~subm.FVC1.isnull()].head(10)

    subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
    sigma_mean = 60
    if sigma_mean<sigma_mean:
        subm['Confidence'] = sigma_opt
    else:
        subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']

    otest = pd.read_csv(PATIENT_DATA)
    for i in range(len(otest)):
        subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
        subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

    subm[["Patient_Week","FVC","Confidence"]].to_csv(QUANTIAL_SUBMISSION, index=False)

    ########################################################################
    #############           SAVE LAST PREDICTION                ############
    ########################################################################

    reg_sub = subm[["Patient_Week","FVC","Confidence"]].copy()
    df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    df = df1[['Patient_Week']].copy()
    df['FVC'] = (0.45*df1['FVC'] + 0.55*df2['FVC'])
    df['Confidence'] = (0.45*df1['Confidence'] + 0.55*df2['Confidence'])
    df.to_csv(FINAL_SUBMISSION, index=False)
