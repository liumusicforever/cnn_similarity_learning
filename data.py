import os
import random

import cv2
import numpy as np
import skimage
import skimage.io
import skimage.transform

class comparsions_iter:
    def __init__(self , root , batch_size):
        self.image_list = get_images(root) 
        self.batch_size = batch_size
        self.cur_idx = 0
        self.num_samples = len(self.image_list)
    def __iter__(self):
        return self
    def next(self):
        if self.cur_idx >= self.num_samples:
            raise StopIteration
        
        if (self.cur_idx + self.batch_size) >= self.num_samples :
            samples = self.image_list[self.cur_idx:]
            self.cur_idx = self.num_samples
        else:
            samples = self.image_list[self.cur_idx:self.cur_idx + self.batch_size]
            self.cur_idx += self.batch_size
        
        comparisions_batch = []
        clssname_batch = []

        for img_path , clss in samples:
            img = load_image(img_path)
            comparisions_batch.append(img)
            clssname_batch.append(clss)

        return [comparisions_batch , clssname_batch]

        

class data_sampler:
    '''
    Structure of dataset:
    Root dir
        |__class_1
            |__xxx.jpg
            |__xxx.jpg
        |__class_2
        |__class_3    
    '''
    def __init__(self,root,batch_size,num_iteration):
       self.dataset = gen_db(root) 
       self.batch_size = batch_size
       self.num_iteration = num_iteration
       self.cur_iteration = 0
    def __iter__(self):
        return self

    def next(self):
        if self.cur_iteration >= self.num_iteration:
            raise StopIteration
        db = self.dataset
        num_classes = len(db.keys())
        
        num_poses = random.randint(1, self.batch_size/2)
        # num_poses = self.batch_size / 2
        num_negs = self.batch_size - num_poses


        # random choice class
        clss_idx = random.sample(range(num_classes), 1)[0]
        clss_name = db.keys()[clss_idx]

        # random choice positive samples
        if num_poses > len(db[clss_name]):
            # if number of positive batch bigger then number of samples ; then append 0:pos sample
            pos_idxes = [i for i in range(len(db[clss_name]))] + [0 for i in range(num_poses - len(db[clss_name]))]
        else:
            pos_idxes = random.sample(range(len(db[clss_name])), num_poses)
        pos_selected = [[db[clss_name][i],1.0] for i in sorted(pos_idxes)]

        # random choice negative samples
        neg_clsses = [db.keys()[idx] for idx in range(num_classes) if idx is not clss_idx]
        neg_samples = []
        for neg_clss in neg_clsses:
            neg_samples += db[neg_clss]
        
        neg_idxes = random.sample(range(len(neg_samples)), num_negs)
        neg_selected = [[neg_samples[i],0.0] for i in sorted(neg_idxes)]

        # shuffle samples
        samples = pos_selected[1:] + neg_selected
        random.shuffle(samples)
        samples = pos_selected[0:1] + samples


        # package to batch
        images_batch = []
        labels_batch = []
        for image_path , clss in samples:
            img = load_image(image_path , shift = True)
            # import matplotlib.pyplot as plt
            # plt.imshow(img)
            # plt.show()
            images_batch.append(img)
            labels_batch.append([clss])
        
        batch = [images_batch , labels_batch]
        self.cur_iteration += 1
        return batch

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
def get_images(root):
    # get all image and their parent dir name
    image_list = []
    for root , subdir , files in os.walk(root):
        for img_filename in files:
            if '.jpg' in img_filename :
                image_path = root+'/'+img_filename
                
                clss = root.split('/')[-1]
                if os.path.exists(image_path):
                    image_list.append([image_path,clss])
                    
    return image_list
def gen_db(root):
    image_list = get_images(root)
    dataset = dict()
    for i , (image_path,clss) in enumerate(image_list):
        if clss  in  dataset:
            dataset[clss].append(image_path)
        else:
            dataset.update({clss:[image_path]}) 
    return dataset
def load_image(path, shift = False):
    # load image
    img = skimage.io.imread(path)

    if shift:
        angle_idx = random.choice([0,1,2,3])
        img = rotate_bound(img,angle_idx*90)

        # angle =random.uniform(0, 1) * 360
        # img = rotate_bound(img,angle)

    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    if shift:
        x_off = random.uniform(0.90, 1.1)
        y_off = random.uniform(0.90, 1.1)
        yy = max(int((img.shape[0] - short_edge * y_off) / 2),0)
        xx = max(int((img.shape[1] - short_edge * x_off) / 2),0)
    else:
        yy = int((img.shape[0] - short_edge ) / 2)
        xx = int((img.shape[1] - short_edge ) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img