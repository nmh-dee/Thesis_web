from django import forms
from .models import AbnPatient

from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.exceptions import ValidationError
from django.core import validators
from .humanize import naturalsize

class CreateForm(forms.ModelForm):
    max_upload_limit = 2*2048 *2048
    max_upload_limit_text = naturalsize(max_upload_limit)

    xray = forms.FileField(required= False, label = 'Upload an Xray image need to predict')
    upload_field_name ='xray'
    class Meta:
        model = AbnPatient
        fields = ['name_patient','xray'] 
    def save(self, commit = True):
        instance = super(CreateForm, self).save(commit = False)
        f = instance.xray #make a copy
        if isinstance(f, InMemoryUploadedFile):
            bytearr = f.read()
            instance.content_type = f.content_type
            instance.xray = bytearr
        if commit:
            instance.save()
        return instance
    