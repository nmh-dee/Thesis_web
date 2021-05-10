import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import csv

import sys
import cv2
import pandas as pd 
import torch
from .retinanet import model
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from .retinanet import csv_eval
from django.conf import settings

from .retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer

# class TempModel(nn.Module):
#     def __init__(self):
#         self.conv1 = nn.Conv2d(3, 5, (3, 3))
#     def forward(self, inp):
#         return self.conv1(inp)


class AbnDetect():
    def __init(self, image_path,model_path, class_list):
        self.image_path = image_path
        self.model_path = model_path
        self.class_list = class_list


    def load_classes(self,csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1
            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = int(class_id)

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result
    #DRAW ANCHOR BOX
    def count_IoU(self, first_box,second_box):
        ixmin = max(first_box[0],second_box[0])
        ixmax = min(first_box[2],second_box[2])
        iymin = max(first_box[1],second_box[1])
        iymax = min(first_box[3],second_box[3])
        iw = np.maximum(ixmax - ixmin +1,0)
        ih = np.maximum(iymax - iymin +1,0)
        intes = iw*ih
        iou = intes/((first_box[2]-first_box[0]+1)*(first_box[3]-first_box[1]+1)+(second_box[2]-second_box[0]+1)*(second_box[3]-second_box[1]+1)-intes)
        return iou
    def clear_data(self, dict_of_box):
        final_dict={}
        for k in dict_of_box:
            output_box_list=[]
            if len(dict_of_box[k])>1:
                clear_box_list=[]
                for first_box in range(len(dict_of_box[k])):
                    if dict_of_box[k][first_box] not in clear_box_list:
                        last_box=dict_of_box[k][first_box]
                    clear_box_list.append(dict_of_box[k][first_box])
                    number_of_overlap=1
                    for second_box in range(len(dict_of_box[k])):
                        if (dict_of_box[k][second_box] not in clear_box_list)&(self.count_IoU(last_box,dict_of_box[k][second_box]) > 0.3):
                            xmin_new = (last_box[0]*number_of_overlap+dict_of_box[k][second_box][0])/(number_of_overlap+1)
                            ymin_new = (last_box[1]*number_of_overlap+dict_of_box[k][second_box][1])/(number_of_overlap+1)
                            xmax_new = (last_box[2]*number_of_overlap+dict_of_box[k][second_box][2])/(number_of_overlap+1)
                            ymax_new = (last_box[3]*number_of_overlap+dict_of_box[k][second_box][3])/(number_of_overlap+1)
                            last_box = [xmin_new,ymin_new,xmax_new,ymax_new]
                            number_of_overlap+=1
                            clear_box_list.append(dict_of_box[k][second_box])
                    if last_box not in output_box_list:
                        output_box_list.append(last_box)
                        if k not in final_dict:
                            final_dict[k]=[last_box]
                        else:
                            final_dict[k].append(last_box)
            else:
                final_dict[k]=[dict_of_box[k][0]]
        return final_dict
    def clean_anchor_box(self,df,img_name,predicted_img,orig_img):
        pred_img_x , pred_img_y , _ = predicted_img.shape
        orig_img_x , orig_img_y , _ = orig_img.shape

        df_find=df[(df.image_id == img_name) & (df.class_id != 14)]
        if len(df_find) > 0 :
            annotations={}
            for index, row in df_find.iterrows():
                row[4] = (row[4]/orig_img_x)*pred_img_x
                row[5] = (row[5]/orig_img_y)*pred_img_y
                row[6] = (row[6]/orig_img_x)*pred_img_x
                row[7] = (row[7]/orig_img_y)*pred_img_y
                if row[1] not in annotations:
                    annotations[row[1]]=[[row[4],row[5],row[6],row[7]]]
                else:
                    annotations[row[1]].append([row[4],row[5],row[6],row[7]])
            bbox_after_clear=self.clear_data(annotations)
            
            txt_name = img_name+".txt"
            for clas in bbox_after_clear:
                for box in bbox_after_clear[clas]:
                    if (clas == "Pneumothorax") | (clas == "Cardiomegaly") | (clas == "ILD") | (clas == "Atelectasis") | (clas == "Aortic enlargement") | (clas == "Infiltration") | (clas == "Consolidation") | (clas == "Lung Opacity") | (clas == "Other lesion") | (clas == "Pulmonary fibrosis"):
                        x1, y1, x2, y2  = box[0],box[1],box[2],box[3]
                        class_id=clas
                        cv2.rectangle(predicted_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,255,0), thickness=2)
                        self.draw_caption(predicted_img,(x1,y1,x2,y2),clas)
        return predicted_img
            
    # Draws a caption above the box in an image
    def draw_caption(self, image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    #VISUALIZE THE DETECTION 
    def detect_image(self):
        df = pd.read_csv(os.path.join(settings.MODEL_ROOT,"train_downsampled.csv"))
        with open(self.class_list, 'r') as f:
            classes = self.load_classes(csv.reader(f, delimiter=','))
        labels = {}
        for key, value in classes.items():
            labels[value] = key
        # retinanet = TempModel()
        retinanet = model.resnet50(num_classes=14, pretrained=True)
        retinanet.load_state_dict(torch.load(self.model_path,map_location=torch.device('cpu')))

        # if torch.cuda.is_available():
        #     model = model.cuda()

        retinanet.training = False
        retinanet.eval()

        image = cv2.imread(self.image_path)
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()
            st = time.time()
            # print(image.shape, image_orig.shape, scale)
            # scores, classification, transformed_anchors = retinanet(image.cuda().float())
            scores, classification, transformed_anchors = retinanet(image.float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                # print(bbox, classification.shape)
                if (label_name == "Pneumothorax") | (label_name == "Cardiomegaly") | (label_name == "ILD") | (label_name == "Atelectasis") | (label_name == "Aortic enlargement") | (label_name == "Infiltration") | (label_name == "Consolidation") | (label_name == "Lung Opacity") | (label_name == "Other lesion") | (label_name == "Pulmonary fibrosis"):
                    score = scores[j]
                    caption = '{} {:.3f}'.format(label_name, score)
                    # draw_caption(img, (x1, y1, x2, y2), label_name)
                    self.draw_caption(image_orig, (x1, y1, x2, y2), caption)
            
                    image_orig = cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            image2 = cv2.imread(self.image_path)
            img_n = self.image_path.split('\\')[-1]
            img_name = img_n.split('_')[0]
            print("image name",img_name)
            last_img =self.clean_anchor_box(df, img_name,image_orig, image2)
            return last_img

'''
detect = AbnDetect()
detect.image_path = 'single_image'
detect.model_path =  'csv_retinanet_106.pt'
detect.class_list = 'class_list.csv'
detect.detect_image()

for image in os.listdir(os.path.join(settings.MODEL_ROOT,"test")):
    if img_name == image:
        orig_img=cv2.imread(os.path.join(os.path.join(settings.MODEL_ROOT,"test/"),image))
self.clean_anchor_box(df,img_name[:-4],image_orig,orig_img)
'''