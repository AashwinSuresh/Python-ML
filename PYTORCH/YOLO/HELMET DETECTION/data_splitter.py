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

images =[
    file for file in os.listdir(image_dir)
    if file.lower().endswith('png')
]

random.shuffle(images)

index = int(len(images)*ratio)
train_images = images[:index]
test_images = images[index:]

print(f"Total Images : {len(images)}")
print(f"Training     : {len(train_images)}")
print(f"Validation   : {len(test_images)}")

def copy_dataset(image_list,image_destination,label_destination):
    for image in tqdm(image_list):
        image_source = os.path.join(image_dir,image)

        label_name = os.path.splitext(image)[0] + '.txt'
        label_source = os.path.join(label_dir,label_name)
        if not os.path.exists(label_source):
            print(f"Missing label for {image}")
            continue
        shutil.copy2(image_source,image_destination)
        shutil.copy2(label_source,label_destination)


print('\nCOPYING TRAINING DATASET ...')
copy_dataset(train_images,train_image_dir,train_label_dir)
print('\nCOPYING TESTING DATASET ...')
copy_dataset(test_images,test_image_dir,test_label_dir)

print("\nDataset split completed successfully.")

