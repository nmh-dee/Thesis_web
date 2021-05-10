# Generated by Django 3.1.7 on 2021-05-07 16:31

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('pfds', '0004_auto_20210507_1219'),
    ]

    operations = [
        migrations.AddField(
            model_name='ppatient',
            name='relative_value',
            field=models.FloatField(blank=True, max_length=6, null=True),
        ),
        migrations.AlterField(
            model_name='ppatient',
            name='gender',
            field=models.CharField(blank=True, max_length=10),
        ),
        migrations.AlterField(
            model_name='ppatient',
            name='smoke',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.CreateModel(
            name='FVC',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('week', models.IntegerField(max_length=3)),
                ('fvc_value', models.FloatField(max_length=6)),
                ('ppatient_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='pfds.ppatient')),
            ],
        ),
    ]