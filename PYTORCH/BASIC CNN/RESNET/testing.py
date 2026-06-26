from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time

def main():
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),      #standard for resnet (since it is used to this format) , but the value can be anythin 256x256
            transforms.ToTensor(),  #converts to tensor , make it into channel x height x width , also divides by 255 
            transforms.Normalize(
                mean =[0.485,0.456,0.406],      #these values are standard values which are used while training and creating resnet 
                std  =[0.229,0.224,0.225]
            )
    ])

    test_dataset = ImageFolder (
        root=r'dataset\test',
        transform=test_transform,
    )

    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers = 4,
            pin_memory=True ,   #instead of copying the converted image to temporary memory , it copies to a pinned memory which makes it easier for the gpu to acess thus making it faster
            persistent_workers=True # instead of destroying workers after each epoch adn then creating it again ,  this one keeps the workers active thus reducing the overhead in creating workers in each epoch
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features,2)
    model.load_state_dict(torch.load(r'..\models\resnet_cnn.pth',weights_only=True))
    model = model.to('cuda')

    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
            acc = 0
            count = 0
            correct = 0
            total = 0
            running_loss = 0
            for images,labels in test_loader:
                images = images.to('cuda',non_blocking = True)
                labels = labels.to('cuda',non_blocking = True)

                outputs = model(images)

                loss = loss_fn(outputs,labels)

                predictions = torch.argmax(outputs,dim=1)
            
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                running_loss+=loss.item()

    acc = 100 * correct / total
    loss_val = running_loss/len(test_loader)  
    end_time = time.perf_counter()
    tot_time = end_time-start_time
    print("+-------------------------+")
    print(f"|{'TESTING':<25}|")
    print("+-------------------------+")
    print(f"|{f'total_inputs : {total}':<25}|")
    print(f"|{f'correct : {correct}':<25}|")
    print(f"|{f'accuracy : {acc:.4f}':<25}|")
    print(f"|{f'loss : {loss_val:.4f}':<25}|")
    print(f"|{f'time_taken : {tot_time:.2f}s':<25}|")
    print("+-------------------------+")

if __name__ =='__main__':
    main()