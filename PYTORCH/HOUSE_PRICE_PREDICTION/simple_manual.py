import torch
import torch.nn as nn
import torch.optim 
from tabulate import tabulate


N=1000
x_train = (torch.randn(N,3)).to('cuda')*1000 #this defines the price1,price2,price3


w_true = (torch.tensor([[20.0],[10.0],[20.0]])).to('cuda')


b_true = (torch.tensor([[20]])).to('cuda')



y_target = x_train @ w_true + b_true



model = nn.Sequential(
    nn.Linear(3,1),
)

print(f"initial weight and bias : {model[0].weight.tolist(),model[0].bias.item()}")
model.to("cuda")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.01
)

loss_fn = nn.MSELoss()

epochs = 30000

print("+------------------------------+------------------------------+")
print(f"|{'EPOCHS':<30}|{'LOSS':<30}|")
print("+------------------------------+------------------------------+")

for epoch in range(epochs):
    y_pred = model(x_train)

    loss = loss_fn(y_pred,y_target)

    optimizer.zero_grad()   #clears out the weight gradient and bias gradient , it is crucial or else on each iteration the gradients might add up

    loss.backward() # calculates grad for weights and bias

    optimizer.step()

    if epoch % 1000 == 0:
        print(f"|{epoch:<30}|{round(loss.item(),2):<30}|")

print("+------------------------------+------------------------------+")


print(f"actual weight : {w_true.tolist()} , predicted weight : {model[0].weight.tolist()}")
print(f"actual bias : {b_true.item()} , predicted bias : {model[0].bias.item()}")
