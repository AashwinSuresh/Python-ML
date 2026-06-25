import torch
import numpy as np
import time
import torch.nn as nn
from mnist_training import CNN
from sklearn.metrics import accuracy_score
def main():
    ###############
    #   TESTING   #
    ###############

    loss_fn =nn.CrossEntropyLoss()
    model = CNN()
    model.load_state_dict(torch.load(r'models\mnist_cnn.pth'))
    model.to('cuda')

    images = np.load(r'.\data\test_images.npy')
    labels = np.load(r'.\data\test_labels.npy')

    images = torch.tensor(images).float()/255.0
    images = images.unsqueeze(1).to('cuda')
    labels = torch.tensor(labels).to('cuda')

    start_time = time.perf_counter()

    model.eval()    #to deactivate the dropout layer

    with torch.no_grad():
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
            