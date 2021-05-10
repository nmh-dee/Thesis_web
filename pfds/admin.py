from django.contrib import admin
from .models import PPatient,PPatientImages
# Register your models here.

admin.site.register(PPatientImages)
admin.site.register(PPatient)