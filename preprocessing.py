import os
import json
import sys
import xmltodict
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random
import cv2
import _pickle as cPickle

DATASET = 'VOC2012/' # original VOC dataset
ANNOTATIONS = os.path.join(DATASET, 'Annotations') # xml annotations
IMAGESPATH = os.path.join(DATASET,'JPEGimages') # images
TRAIN = 'TrainData'
VAL = 'ValidationData'

CLASSES = {
    'aeroplane'   : 0,
    'bicycle'     : 1,
    'bird'        : 2,
    'boat'        : 3,
    'bottle'      : 4,
    'bus'         : 5,
    'car'         : 6,
    'cat'         : 7,
    'chair'       : 8,
    'cow'         : 9,
    'diningtable' : 10,
    'dog'         : 11,
    'horse'       : 12,
    'motorbike'   : 13,
    'person'      : 14,
    'pottedplant' : 15,
    'sheep'       : 16,
    'sofa'        : 17,
    'train'       : 18,
    'tvmonitor'   : 19
}

def xmltojson(xml):
	f = open(xml).read()
	d = xmltodict.parse(f)
	mapping = {}
	mapping['filename'] = d['annotation']['filename']
	mapping['dimensions'] = (int(d['annotation']['size']['width']), 
							 int(d['annotation']['size']['height']), 
							 int(d['annotation']['size']['depth']))
	mapping['boxes'] = []

	if not isinstance(d['annotation']['object'], list):
		d['annotation']['object'] = [d['annotation']['object']]

	for thing in d['annotation']['object']:
		boxdata = {}
		if thing['name'] not in CLASSES:
			continue
		boxdata['category'] = thing['name']
		boxdata['xmin'] = int(float(thing['bndbox']['xmin']))
		boxdata['ymin'] = int(float(thing['bndbox']['ymin']))
		boxdata['xmax'] = int(float(thing['bndbox']['xmax']))
		boxdata['ymax'] = int(float(thing['bndbox']['ymax']))
		mapping['boxes'].append(boxdata)
	return mapping

#print(json.dumps(xmltojson(ANNOTATIONS+'/2007_000032.xml'), indent=3))

def generateData(filename):
	path = VAL if random.random() > 0.85 else TRAIN
	img = Image.open(os.path.join(IMAGESPATH, filename+'.jpg'))
	img = img.resize((448,448), Image.BILINEAR)
	img.save(os.path.join(path,'X', filename+'.jpg'), "JPEG")
	imgdata = xmltojson(os.path.join(ANNOTATIONS, filename+'.xml'))
	imgwratio = 448/imgdata['dimensions'][0] #width   
	imghratio = 448/imgdata['dimensions'][1] #height
	label = np.zeros(shape=(7,7,30))
	for box in imgdata['boxes']:
		w = box['xmax']-box['xmin']
		h = box['ymax']-box['ymin']
		x = box['xmin'] + w//2
		y = box['ymin'] + h//2
		w=int(w*imgwratio)
		h=int(h*imghratio)
		x=int(x*imgwratio)
		y=int(y*imghratio)
		i, j = x//64, y//64 
		w/=448 # sacaled from all picture
		h/=448
		x = (x/64)-i #scaled form grid cell
		y = (y/64)-j
		label[i][j][0:5] = [x,y,w,h,1]
		label[i][j][5:10] = [x,y,w,h,1]
		label[i][j][10+CLASSES[box['category']]]=1
		"""color = (255, 0, 0) 
		thickness = 2
		window_name = 'Test'
		image = cv2.imread(os.path.join(TRAIN,'X', filename+'.jpg'))
		image = cv2.rectangle(image, (x-w//2,y+h//2), (x+w//2,y-h//2), color, thickness) 
		# Displaying the image  
		cv2.imshow(window_name, image)
		cv2.waitKey(0)"""
	cPickle.dump(label, open(os.path.join(path,filename+'.pkl'),'wb'))




		
		



#generateData('2007_000032')

thing_to_save = cPickle.load( open(os.path.join(TRAIN,'2007_000032'+'.pkl'),"rb"))
print(type(thing_to_save))















