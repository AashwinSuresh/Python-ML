import os
import random 
import shutil
from tqdm import tqdm 

random.seed(42)
image_dir = r'data\images'
label_dir = r'data\labels'
ratio = 0.8

train_image_dir = os.path.join(image_dir,'train')
test_image_dir = os.path.join(image_dir,'test')
train_label_dir = os.path.join(label_dir,'train')
test_label_dir = os.path.join(label_dir,'test')

os.makedirs(train_image_dir,exist_ok=True)
os.makedirs(test_image_dir,exist_ok=True)
os.makedirs(train_label_dir,exist_ok=True)
os.makedirs(test_label_dir,exist_ok=True)




