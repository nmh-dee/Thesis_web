from django.db import models

# Create your models here.
from django.db import models
from django.core.validators import MinLengthValidator
from django.conf import settings
# Create your models here.

class PPatient(models.Model):
    id = models.AutoField(primary_key = True)
    name = models.CharField(max_length=200)
    ages = models.DecimalField(max_digits= 3, decimal_places= 0)
    gender = models.CharField(blank= True,max_length= 10)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, default=1, on_delete = models.CASCADE)
    smoke = models.CharField(blank= True, max_length=200,null= True )
    relative_a = models.FloatField(max_length= 6,blank= True, default= '0.5')
    relative_b = models.FloatField(max_length=6, default=0)
    FVC_base = models.FloatField(max_length= 4, default=3000.0)
    week_base = models.IntegerField(default= 0)
    percent = models.FloatField(max_length=4, default= 100)
    week_start = models.IntegerField(default=0)
    week_end = models.IntegerField(default=20 )

class PPatientImages(models.Model):
    id = models.AutoField(primary_key = True)
    ppatient_id = models.ForeignKey(PPatient, on_delete = models.CASCADE)
    image = models.FileField(upload_to='pf/' ,max_length = 255)

class FVC(models.Model):
    id = models.AutoField(primary_key=True)
    ppatient_id = models.ForeignKey(PPatient, on_delete = models.CASCADE)
    week = models.IntegerField()
    fvc_value_e = models.FloatField(max_length=6)
    fvc_value_q = models.FloatField(max_length=6, default= 0)
    fvc_value_final = models.FloatField(max_length=6, default= 0)
    