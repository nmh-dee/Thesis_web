# Generated by Django 3.1.7 on 2021-05-08 17:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pfds', '0005_auto_20210507_2331'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ppatient',
            name='relative_value',
            field=models.FloatField(blank=True, default='0.5', max_length=6),
        ),
    ]
