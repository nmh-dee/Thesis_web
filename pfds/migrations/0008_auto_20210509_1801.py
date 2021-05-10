# Generated by Django 3.1.7 on 2021-05-09 11:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pfds', '0007_auto_20210509_0032'),
    ]

    operations = [
        migrations.RenameField(
            model_name='fvc',
            old_name='fvc_value',
            new_name='fvc_value_e',
        ),
        migrations.RenameField(
            model_name='ppatient',
            old_name='relative_value',
            new_name='relative_a',
        ),
        migrations.AddField(
            model_name='fvc',
            name='fvc_value_final',
            field=models.FloatField(default=0, max_length=6),
        ),
        migrations.AddField(
            model_name='fvc',
            name='fvc_value_q',
            field=models.FloatField(default=0, max_length=6),
        ),
        migrations.AddField(
            model_name='ppatient',
            name='FVC_base',
            field=models.FloatField(default=3000.0, max_length=4),
        ),
        migrations.AddField(
            model_name='ppatient',
            name='relative_b',
            field=models.FloatField(default=0, max_length=6),
        ),
        migrations.AddField(
            model_name='ppatient',
            name='week_base',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='ppatient',
            name='week_end',
            field=models.IntegerField(default=20),
        ),
        migrations.AddField(
            model_name='ppatient',
            name='week_start',
            field=models.IntegerField(default=0),
        ),
    ]
