# Generated by Django 3.1.7 on 2021-05-08 17:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pfds', '0006_auto_20210509_0031'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fvc',
            name='week',
            field=models.IntegerField(),
        ),
    ]
