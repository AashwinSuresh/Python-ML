import xml.etree.ElementTree as et
from tqdm import tqdm
import time 
import os 

tree = et.parse(r'data\annotations\hard_hat_workers0.xml')
root = tree.getroot()
name = root.find('filename').text

annotations=r"data\annotations"
labels = r'data\labels'
os.makedirs(labels,exist_ok=True)

class_map ={
    'helmet' : 0,
    'head' : 1,
    'person' : 2,
}



def convert_xml(input_path,output_path):
    tree = et.parse(input_path)
    root = tree.getroot()
    width = float(root.find(r'size/width').text)
    height = float(root.find(r'size/height').text)
    yolo_op = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.strip().lower()
        if class_name not in class_map:
            os.system('cls')
            print(f"skipping unknown class {class_name} in file {input_path}")
            continue
        class_id = class_map[class_name]
        box = obj.find('bndbox')
        x_min = float(box.find('xmin').text)
        y_min = float(box.find('ymin').text)
        x_max = float(box.find('xmax').text)
        y_max = float(box.find('ymax').text)

        box_height = y_max-y_min
        box_width  = x_max-x_min
        center_x = x_min+(box_width)/2
        center_y = y_min+(box_height)/2

        center_x/=width
        center_y/=height
        box_height/=height
        box_width/=width

        yolo_op.append(
            f'{class_id} '
            f'{center_x:.6f} '
            f'{box_width:.6f} '            
            f'{box_height:.6f} '
        )

       

    with open(output_path,'w') as f:
        f.write("\n".join(yolo_op))

        
        

def main():
    xml_files = [file for file in os.listdir(annotations) if file.endswith('.xml')]
    txt_name = os.path.splitext(xml_files[0])[0] + '.txt'
    converted = 0
    
    for file in tqdm(xml_files,desc='CONVERTING'):
        txt_name = os.path.splitext(file)[0] + '.txt'
        xml_path = os.path.join(annotations,file)
        txt_path = os.path.join(labels,txt_name)
        convert_xml(xml_path,txt_path)
        converted+=1
        # if converted%100 == 0 :
        #     os.system('cls')
        #     print(f"converted = {converted}")

if __name__ == '__main__':
    main()

    

