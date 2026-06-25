import numpy as np
import torch
import time
import torch.optim
import torch.nn as nn
from torch.utils.data import (TensorDataset,DataLoader)
from sklearn.metrics import accuracy_score

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3
                ),                  #outsize = [(insize-kernal +2*padd)/stride]+1
                                    #outsize = 28 - 3 +1 = 26
                
                nn.MaxPool2d(2),    # outsize = insize/2 , since here a 2x2 kernal is used
                                    # outsize = 26/2 = 13

                nn.Conv2d(
                    in_channels = 32,
                    out_channels= 64,
                    kernel_size = 3  #outsize =  13 - 3 +1 = 11
                ),

                nn.MaxPool2d(2),    #outsize = 11//2 = 5        
        )       # in the end of the convolution layers it will output 5*5 image with 64 feature maps

        self.fc_layers = nn.Sequential(
                nn.Linear(64*5*5,260),
                nn.Dropout(0.3),    #30% of hte output will be zero(only during training) , to preven overfitting
                nn.ReLU(),
                nn.Linear(260,128),
                nn.Dropout(0.2),
                nn.GELU(),
                nn.Linear(128,10)
        )

    def forward(self,x):        #the name should be forward , then only model(images) will work
        x = self.conv_layers(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc_layers(x)
        return x
def main():

    ##############
    #  TRAINING  #
    ##############
            
    images = np.load(r'.\data\train_images.npy')
    labels = np.load(r'.\data\train_labels.npy')

    normalized_image = torch.tensor(images).float() / 255.0 
    train_image = normalized_image.unsqueeze(1)
    train_label = torch.tensor(labels)

    train_dataset = TensorDataset(train_image,train_label)

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=6000,
    )
    
    model = CNN().to('cuda')
    
    model.train()   # to enter training mode , activates the dropout layer 

    epochs = 30

    loss_fn = nn.CrossEntropyLoss()     #it already do softmax for you 
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=0.001
    )
    print("+------------------------------+------------------------------+------------------------------+")
    print(f"|{'EPOCHS':<30}|{'LOSS':<30}|{'ACCURACY':<30}|")
    print("+------------------------------+------------------------------+------------------------------+")
    acc=0
    for epoch in range(epochs+1):
        acc = 0
        count = 0
        for image_batch,label_batch in train_loader:
            images = image_batch.to('cuda')
            labels = label_batch.to('cuda')

            optimizer.zero_grad()

            output = model(images)

            loss = loss_fn(output,labels)

            loss.backward()

            optimizer.step()

            predictions = torch.argmax(output,dim=1)

            if epoch%10 == 0:
                count+=1
                acc += accuracy_score(predictions.detach().cpu().tolist(),labels.detach().cpu().tolist())
        if epoch%10 == 0:
            acc = acc/float(count)
            acc*=100    
            print(f"|{epoch:<30}|{round(loss.item(),4):<30}|{round(acc,2):<30}|")
    print("+------------------------------+------------------------------+------------------------------+")
    print(f"\ntraining finished .... \n accuracy obtained : {acc:.3}")

    ###############
    #   TESTING   #
    ###############

    images = np.load(r'.\data\test_images.npy')
    labels = np.load(r'.\data\test_labels.npy')

    images = torch.tensor(images).float()/255.0
    images = images.unsqueeze(1).to('cuda')
    labels = torch.tensor(labels).to('cuda')

    start_time = time.perf_counter()

    model.eval()    #to deactivate the dropout layer

    outputs = model(images)

    loss = loss_fn(outputs,labels)

    predictions = torch.argmax(outputs,dim=1)

    correct  = (predictions == labels).sum().item()
    acc = accuracy_score(predictions.detach().cpu().tolist(),labels.detach().cpu().tolist())

    end_time = time.perf_counter()
    tot_time = end_time-start_time
    print("+-------------------------+")
    print(f"|{'TESTING':<25}|")
    print("+-------------------------+")
    print(f"|{f'total_inputs : {len(predictions.tolist())}':<25}|")
    print(f"|{f'correct : {correct}':<25}|")
    print(f"|{f'accuracy : {acc}':<25}|")
    print(f"|{f'loss : {loss:.4f}':<25}|")
    print(f"|{f'time_taken : {tot_time:.2f}s':<25}|")
    print("+-------------------------+")

if __name__ == '__main__':
    main()           
            









