# Generated by Django 3.1.7 on 2021-05-05 15:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pfds', '0002_auto_20210505_1315'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ppatientimages',
            name='image',
            field=models.FileField(max_length=255, upload_to='pf/'),
        ),
    ]
