from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn 
import torch.optim
from torch.utils.data import DataLoader 

    
def main():
    model =  resnet18(weights='DEFAULT')

    model.fc = nn.Linear(model.fc.in_features,2)

    model = model.to('cuda')

    for param in model.parameters():
        param.requires_grad = False     #because we dont want to change the other parameters of already pretrained model over millions of data
                                        # we only want to change the parameters 
    for param  in model.fc.parameters():
        param.requires_grad = True      # only activate the final classification layer , here it acitavates the in_features and the out_features 

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),      #standard for resnet (since it is used to this format) , but the value can be anythin 256x256
        
        #data augmentation steps    only need while training , not during testing
        # transforms.RandomHorizontalFlip(p=0.5),     #50% prob to flip the image like in a mirror
        # transforms.RandomRotation(10),               #random rotation bw -10deg to 10deg
        # transforms.ColorJitter(
        #     brightness = 0.2,   #converts brightness to bw 1-0.2 and 1+0.2  
        #     saturation = 0.3,
        #     contrast  = 0.2 ,
        # ),

        transforms.ToTensor(),  #converts to tensor , make it into channel x height x width , also divides by 255 

        transforms.Normalize(
            mean =[0.485,0.456,0.406],      #these values are standard values which are used while training and creating resnet 
            std  =[0.229,0.224,0.225]
        )

    ])

    train_dataset = ImageFolder(        #it automaitcally classifies and gives labels by using the folder
        root=r'dataset\train',          # everythin in cats folder is given label 0 and everything in dogs folder is given label 1
        transform=train_transform        
    )

    # print(train_dataset.class_to_idx)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers = 4,
        pin_memory=True ,   #instead of copying the converted image to temporary memory , it copies to a pinned memory which makes it easier for the gpu to acess thus making it faster
        persistent_workers=True # instead of destroying workers after each epoch adn then creating it again ,  this one keeps the workers active thus reducing the overhead in creating workers in each epoch
    )

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),  #since we are only changing the weights of that layer
                                #we can also just add model.parameters() , since there will be no gradient it will automatically skips 
        lr = 0.001                        
    )

    loss_fn = nn.CrossEntropyLoss()

    model.train()

    epochs = 10
    print("+------------------------------+------------------------------+------------------------------+")
    print(f"|{'EPOCHS':<30}|{'LOSS':<30}|{'ACCURACY':<30}|")
    print("+------------------------------+------------------------------+------------------------------+")
    acc=0
    for epoch in range(epochs+1):
        acc = 0
        count = 0
        correct = 0
        total = 0
        running_loss = 0
        for images,labels in train_loader:
            images = images.to('cuda',non_blocking = True)
            labels = labels.to('cuda',non_blocking = True)

            outputs = model(images)

            loss = loss_fn(outputs,labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            predictions = torch.argmax(outputs,dim=1)

            if epoch%1 == 0:
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                running_loss+=loss.item()

        if epoch%1 == 0: 
            acc = 100 * correct / total
            loss_val = running_loss/len(train_loader)  
            print(f"|{epoch:<30}|{round(loss_val,4):<30}|{round(acc,2):<30}|")
    print("+------------------------------+------------------------------+------------------------------+")
    print(f"\ntraining finished .... \n accuracy obtained : {acc:.3f}")

    torch.save(model.state_dict(),r'..\models\resnet_cnn.pth')
    print(f"model saved in folder models")


if __name__ == '__main__':
    main()