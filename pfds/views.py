from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .models import PPatientImages,PPatient, FVC
from .owner import OwnerDetailView, OwnerListView
import os
import pandas as pd
import sys
import csv
import pydicom
from .preprocess import Pre_image, Load_Eff, IGenerator
import numpy as np
from django.conf import settings
import tensorflow as tf
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
from sklearn.metrics import mean_absolute_error

# Create your views here.
'''
class AbnDetailView(OwnerDetailView):
    template_name = 'pfds/pfd_detail.html'
    model = PPatient
    
    

class AbnListView (OwnerListView):
    model = AbnPatient
    template_name = 'pfds/pfd_list.html'
    def get(self, request):
        apatient_list = AbnPatient.objects.all()
        ctx = {'apatient_list':apatient_list}
        return render(request,'abns/abn_list.html',ctx)
'''



def MultipleUpload(request):
    return render(request,"pfds/multiple_fileupload.html")

def Multipleupload_save(request):
    name= request.POST.get("name")
    ages = int(request.POST.get("ages"))
    gender = request.POST.get("gender")
    smoke = request.POST.get("smoke")
    FVC_base = float(request.POST.get("fvc"))
    week_base = int(request.POST.get("week"))
    percent = float(request.POST.get("percent"))
    images = request.FILES.getlist("file[]")
    week_start = int(request.POST.get("week_start"))
    week_end = int(request.POST.get("week_end"))
    ppatient1= PPatient(name=name, ages=ages, gender=gender, smoke=smoke, FVC_base= FVC_base, week_base= week_base, week_start= week_start, week_end= week_end, percent= percent)
    ppatient1.save()
    kwargs = {'newline': ''}
    base_dir = os.path.join(settings.MODEL_ROOT,str(ppatient1.id))
    try:
        os.makedirs(base_dir) # create destination directory, if needed (similar to mkdir -p)
        with open(os.path.join(os.path.join(base_dir),'demo.csv'),'w', **kwargs) as demo:
            writer = csv.writer(demo, delimiter=',')
            writer.writerow(['Patient','Weeks','FVC', 'Percent','Age','Sex','SmokingStatus'])
            writer.writerow([ppatient1.name, ppatient1.week_base, ppatient1.FVC_base, ppatient1.percent,ppatient1.ages,ppatient1.gender,ppatient1.smoke])
        with open(os.path.join(base_dir,'efficient_sub.csv'),'w', **kwargs ) as ef:
            ef.write('')
        with open(os.path.join(base_dir,'quantial_sub.csv'), 'w', **kwargs) as qt:
            qt.write('')
        with open(os.path.join(base_dir,'final_sub.csv'), 'w', **kwargs) as final:
            final.write('')
    except OSError:
    # The directory already existed, nothing to do
        pass
    for img in images:
        fs= FileSystemStorage()
        file_path= fs.save(img.name, img)
        pimage = PPatientImages(ppatient_id= ppatient1, image=file_path)
        pimage.save()
    pp_imgs = PPatientImages.objects.filter(ppatient_id = ppatient1.id)
    #Processing ---------------
    '''
    x=[]
    tab =[]
    pre = Pre_image()
    pre.my_shape = (64,64)
    slices = [pydicom.dcmread(pp_img.image.path) for pp_img in pp_imgs]
    hu_scans= pre.transform_to_hu(slices)
    
    prepared_scans = np.zeros((hu_scans.shape[0], pre.my_shape[0], pre.my_shape[1]), dtype= np.int16)
    hu_scans= hu_scans.astype(np.int32)
    print(hu_scans.shape)
    for s in range(hu_scans.shape[0]):
        prepared_scans[s]= pre.resize_scan(scan= hu_scans[s,:,:])
    if hu_scans.shape[1]!= hu_scans.shape[0]:
        hu_scans = hu_scans.astype(np.int32)
        for s in range(hu_scans.shape[0]):
            if pre.my_shape[0]==512:
                prepared_scans[s] = pre.crop_scan(hu_scans[s,:,:])
            else:
                prepared_scans[s] = pre.crop_and_resize(hu_scans[s,:,:])

    image_segmented = pre.segment_lung_mask(prepared_scans)
    x=[]
    for dim in range(image_segmented.shape[0]):
        x.append(image_segmented[dim])
        tab.append(IGenerator.get_tab(ppatient1))
    x = np.expand_dims(x, axis=-1)
    tab = np.array(tab)
    e = Load_Eff(weights_file= os.path.join(settings.MODEL_ROOT,'effnet_30.h5'),shape = (64,64,1))
    emodel = e.build_model()
    _a = emodel.predict([x, tab])
    a = np.quantile(_a, 0.5)
    '''
    ppatient1.relative_a = -7.9015
    #ppatient1.relative_b = ppatient1.FVC_base - a*ppatient1.week_base
    ppatient1.relative_b = 3379.6362
    #-------------------------------------------------------------
    data =[]
    #demo = pd.read_csv("demo.csv")
    for i in range(int(week_start),int(week_end)):
        info = str(ppatient1.name) + '_' +str(i)

        data.append(str(info))
        # print('data:', data)
    # print('data_all:', data)
    kwargs = {'newline': ''}
    mode = 'w'
    if sys.version_info<(3,0):
        kwargs.pop('newline', None)
        mode = "wb"
    with open(os.path.join(base_dir,'submission.csv'), mode, **kwargs) as fp:
        writer = csv.writer(fp)
        writer.writerow(["Patient_Week", "FVC", "Confidence"])  # write header
        for i in range(len(data)):
            writer.writerow([data[i]])
        # writer.writerow([data[1]])
    with open(os.path.join(base_dir,'submission_copy.csv'), mode,**kwargs) as fp:
        writer = csv.writer(fp)
        writer.writerow(["Patient_Week", "FVC", "Confidence"])
        for i in range(len(data)):
            writer.writerow(data[i])
    #--------------------------------------------------------
    subs=[]
    sub = pd.read_csv(os.path.join(base_dir,'submission.csv'))
    for k in sub.Patient_Week.values:
        print('k', k)
        p, w = k.split('_')
        print('w',w)
        w = int(w)
        print('a:', ppatient1.relative_a)
        print('b:', ppatient1.relative_b)
        fvc = ppatient1.relative_a*w + ppatient1.relative_b
        sub.loc[sub.Patient_Week==k, 'FVC'] = fvc
        sub.loc[sub.Patient_Week == k, 'Confidence'] =(float(ppatient1.percent)- ppatient1.relative_a*abs(ppatient1.week_base- w))
    _sub = sub[["Patient_Week","FVC","Confidence"]].copy()
    subs.append(_sub)

    #---------------------------------------------------------
    #save efficient net predict
    N = len(subs)
    sub = subs[0].copy()
    sub["FVC"] = 0
    sub["Confidence"] = 0
    for i in range(N):
        sub["FVC"] += subs[0]["FVC"] * (1/N)
        sub["Confidence"] += subs[0]["Confidence"] * (1/N)
    sub[["Patient_Week","FVC","Confidence"]].to_csv(os.path.join(base_dir, 'efficient_sub.csv'), index=False)
    img_sub = sub[["Patient_Week","FVC","Confidence"]].copy()
    #build quantial net
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


    #load quantile net
    tr = pd.read_csv(os.path.join(settings.MODEL_ROOT,'train.csv'))
    tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
    
    
    chunk = pd.read_csv(os.path.join(base_dir,"demo.csv"))
    sub =pd.read_csv(os.path.join(base_dir, 'efficient_sub.csv'))
    #demo= PATIENT_DATA = os.path.join(settings.MODEL_ROOT,str(ppatient1.name)+".csv")
    sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
    sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
    sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
    sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
    #sub = pd.concat(sub,chunk.drop('Weeks', axis =1), )
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
    print('FE', FE)

    tr = data.loc[data.WHERE=='train']
    chunk = data.loc[data.WHERE=='val']
    sub = data.loc[data.WHERE=='test']
    del data

    y = tr['FVC'].values
    z = tr[FE].values
    ze = sub[FE].values
    print(ze.shape)
    nh = ze.shape[1]
    print('nh', nh)
    pe = np.zeros((ze.shape[0], 3))
    pred = np.zeros((z.shape[0], 3))

    net = make_model(nh)
    net.load_weights(os.path.join(settings.MODEL_ROOT, 'savelastmodel.h5'))
    pe = net.predict(ze, batch_size=128, verbose=0) #/ 2 #NFOLD in training process is 2
    pred = net.predict(z, batch_size=128, verbose=0)
    # qmodel = Load_Qt(weights_file = os.path.join(settings.MODEL_ROOT, 'savelastmodel.h5'))
    # qnet = qmodel.make_model(nh= nh)
    # qnet.load_weights()

    #save quantial predict
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

    otest = pd.read_csv(os.path.join(base_dir,'demo.csv'))
    for i in range(len(otest)):
        subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
        subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

    subm[["Patient_Week","FVC","Confidence"]].to_csv(os.path.join(base_dir,'quantial_sub.csv'), index=False)
    #final
    reg_sub = subm[["Patient_Week","FVC","Confidence"]].copy()
    df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    df = df1[['Patient_Week']].copy()
    df['FVC'] = (0.45*df1['FVC'] + 0.55*df2['FVC'])
    df['Confidence'] = (0.45*df1['Confidence'] + 0.55*df2['Confidence'])
    df.to_csv(os.path.join(base_dir, 'final_sub.csv'), index=False)

    return HttpResponse("predicted")

