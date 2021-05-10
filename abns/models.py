from django.db import models

# Create your models here.
from django.core.validators import MinLengthValidator
from django.conf import settings

# Create your models here.
class AbnPatient(models.Model):
    name_patient = models.CharField(max_length=200,validators=[MinLengthValidator(2, "Name must be greater than 2 characters")]
    )

    owner = models.ForeignKey(settings.AUTH_USER_MODEL,default=1,on_delete= models.CASCADE)
    xray = models.ImageField(null= True,upload_to='xray/', editable=True)
    xray_truth = models.ImageField(null = True,blank = True, upload_to= 'truth_xray/' ,editable= True, help_text ='Upload truth Xray image')
    xray_predicted = xray_predicted = models.CharField(max_length=200, null= True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    content_type = models.CharField(max_length= 256, null= True, help_text= 'The MIMEType of the xray file')
    valuate = models.CharField(max_length= 200, null = True)
    def __str__(self):
        return self.name_patient