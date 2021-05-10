from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .models import AbnPatient
from .owner import OwnerDeleteView, OwnerDetailView, OwnerListView, OwnerUpdateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy, reverse
from django.views import View
from .form import CreateForm
from django.http import HttpResponse
from django.conf import settings
import cv2
import numpy as np
import os
from PIL import Image
from .retina_detect import AbnDetect

import torch

# Create your views here.
class AbnDetailView(OwnerDetailView):
    template_name = 'abns/abn_detail.html'
    model = AbnPatient
    
    

class AbnListView (OwnerListView):
    model = AbnPatient
    template_name = 'abns/abn_list.html'
    def get(self, request):
        apatient_list = AbnPatient.objects.all()
        ctx = {'apatient_list':apatient_list}
        return render(request,'abns/abn_list.html',ctx)


class AbnUpdateView(LoginRequiredMixin,View):
    template_name = 'abns/abn_form.html'
    success_url = reverse_lazy('abns:abn_detail')

class AbnDeleteView(OwnerDeleteView):
    model = AbnPatient
    template_name = 'abns/abnpatient_corfirm_deltete.html'

class AbnCreateView(LoginRequiredMixin, View):
    template_name = 'abns/abn_form.html'
    success_predict = 'abns/abn_detail.html'
    def get(self, request, pk= None):
        form = CreateForm
        ctx = {'form':form}
        return render(request, self.template_name,ctx)
    def post(self, request, pk= None):
        form = CreateForm(request.POST, request.FILES or None)
        if not form.is_valid():
            ctx ={'form':form}
            return render(request,self.template_name,ctx)
        abnpatient= form.save(commit= True)


        abnpatient.owner = self.request.user
        image_path = abnpatient.xray.path
        detect = AbnDetect()
        detect.image_path = os.path.join(settings.MEDIA_ROOT,abnpatient.xray.path)
        detect.model_path =  os.path.join(settings.MODEL_ROOT,'csv_retinanet_106.pt')
        detect.class_list = os.path.join(settings.MODEL_ROOT, 'class_list.csv')
        img = detect.detect_image()
        abnpatient.xray_predicted = abnpatient.xray.name.replace("xray","xray_predicted")
        print("xray_predicted_url:",abnpatient.xray_predicted  )
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, abnpatient.xray_predicted), img)
        abnpatient.save()
        ctx ={'abnpatient':abnpatient}
        return render(request, self.success_predict, ctx)


