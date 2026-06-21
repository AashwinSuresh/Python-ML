from tabulate import tabulate
from termcolor import colored
import torch

##############################
# IMPLEMENTING TRAINGIN DATA #
##############################

#each datapoints (x,y) have one feature , d_in = 3 and d_out = 2 means :
# y1 = w11*x1 + w12*x2 + w13*x3 + b1
# y2 = w21*x1 + w22*x2 + w23*x3 + b2

d_in = 1
d_out = 1
N = 10   #no of train data
x_train = torch.randn(N,d_in)
W_TRUE = torch.tensor([[2.0]])
B_TRUE = torch.tensor(3.0)
noise = torch.randn(N,d_out)
y_true = x_train@W_TRUE + B_TRUE
print(f"TRUE WEIGHT : {W_TRUE.item()} \n TRUE BIAS : {B_TRUE.item()}")



##################
# MODEL TRAINING #
##################

W = torch.randn(d_in,d_out,requires_grad=True)
B = torch.randn(1,requires_grad=True)

print(f"\ninitial weight : {W.item()} \n initial bias : {B.item()}\n")
learning_rate =0.01
epochs = 1000

t_epoch = []
t_loss = []
t_w = []
t_b = []
w_ac,w_pred,b_ac,b_pred = 0,0,0,0
for epoch in range(epochs):
    y_train = x_train@W + B 
    loss = torch.mean((y_true-y_train)**2)

    loss.backward() #backward pass to update the weights 

    with torch.no_grad():   #no need to track this in the computational graph
        W-=learning_rate*W.grad
        B-=learning_rate*B.grad

        # 1. Create clean, perfectly padded base strings first (width 7)
        w_true_raw = f"{W_TRUE.item():.1f}".ljust(12)
        w_pred_raw = f"{W.item():.3f}".ljust(12)
        
        b_true_raw = f"{B_TRUE.item():.1f}".ljust(12)
        b_pred_raw = f"{B.item():.3f}".ljust(12)

        # 2. Check distance and apply color to the ALREADY padded strings
        if abs(W_TRUE.item() - W.item()) < 0.3:
            w_ac = colored(w_true_raw, "blue")
            w_pred = colored(w_pred_raw, "blue")
        else:
            w_ac = w_true_raw
            w_pred = w_pred_raw
            
        if abs(B_TRUE.item() - B.item()) < 0.3:
            b_ac = colored(b_true_raw, "blue")
            b_pred = colored(b_pred_raw, "blue")
        else:
            b_ac = b_true_raw
            b_pred = b_pred_raw
    
    W.grad.zero_()
    B.grad.zero_()


    # used to reset the values of the computational graph of W and B , why is it that ? 
    #  w_old
    #  ↓
    #  y1
    #  ↓
    # loss1
    #  ↓
    # update
    #  ↓
    # w_new
    #  ↓
    #  y2
    #  ↓
    # loss2

    # here on the next iteration if we do loss.backward() the gradient we will get is:
    #  d(loss2)/d(w_old)
    #but what we need is d(loss2)/d(w_new)
    
    if epoch%100 == 0:
        t_epoch.append(epoch)
        t_loss.append(f"{loss:.3f}")
        t_w.append(f"{w_ac}|{w_pred}")
        t_b.append(f"{b_ac}|{b_pred}")

headers = ["EPOCH","CALCULATED LOSS","ACTUAL AND PREDICTED WEIGHT","ACTUAL AND PREDICTED BIAS"]
data = zip(t_epoch,t_loss,t_w,t_b)

table = tabulate(data,headers,"grid")


print("\n TABLE ")
print("-------\n")
print(table)
print(f"w true value : {W_TRUE.item()} , w predicted value : {W.item()}")
print(f"b true value : {B_TRUE.item()} , b predicted value : {B.item()}")

    
        


