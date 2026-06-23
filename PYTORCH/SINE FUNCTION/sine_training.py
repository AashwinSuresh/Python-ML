import torch
import warnings
import time
import torch.nn as nn 
import gradio as gr
import torch.optim
import time as t
from torch.utils.data import (TensorDataset,DataLoader)
import matplotlib.pyplot as plt

a=[1,2,3,4]
# def run_without_batching(model,optimizer,loss_fn,x_train,y_train_target):
#     epochs = 1000
#     print("+------------------------------+------------------------------+")
#     print(f"|{'EPOCHS':<30}|{'LOSS':<30}|")
#     print("+------------------------------+------------------------------+")

#     start_time = t.perf_counter()
#     for epoch in range(1,epochs+1):     #it performs 1000 updates (1 in each epoch , and there are 1000 epochs)
#         y_pred = model(x_train)

#         loss = loss_fn(y_pred,y_train_target)

#         optimizer.zero_grad()

#         loss.backward()

#         optimizer.step()

#         if epoch % 10 == 0:
#             print(f"|{epoch:<30}|{round(loss.item(),2):<30}|")
#     print("+------------------------------+------------------------------+")
#     end_time = t.perf_counter()
#     tot_time = end_time - start_time
#     print(f"\nexecution time without batching : {tot_time:.2f}")

def run_with_batching(model,optimizer,loss_fn,x_train,y_train_target):


    dataset = TensorDataset(
        x_train,
        y_train_target    
    )  
    loader = DataLoader(
            dataset=dataset,
            batch_size=100,     #1000 data divided to 10 parts of 100 size , so in each epoch 10 updates will happen
    )


    print("+------------------------------+------------------------------+")
    print(f"|{'EPOCHS':<30}|{'LOSS':<30}|")
    print("+------------------------------+------------------------------+")

    epochs = 1000

    ########
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()

    sin_real,=ax1.plot([],[],linestyle='--',label='Sin Curve')         
    sin_pred,=ax1.plot([],[],color='red',label='Predicted Sin Curve')

    cos_real,=ax3.plot([],[],linestyle='--',label='Cos Curve')
    cos_pred,=ax3.plot([],[],color='red',label='Predicted Cos Curve')

    loss_gr, = ax2.plot([],[],color='blue',label="Loss Calculated")
    ax1.legend()
    ax2.legend()

    ax1.set_xlim(x_train.min().item(),x_train.max().item())
    ax1.set_ylim(-1,1)

    ax2.set_xlim(0,epochs+2)
    ax2.set_ylim(0,0.7)

    ax3.set_xlim(x_train.min().item(),x_train.max().item())
    ax3.set_ylim(-1,1)

    sin_real.set_data(x_train.detach().cpu().numpy(),y_train_target[:,0].detach().cpu().numpy())
    cos_real.set_data(x_train.detach().cpu().numpy(),y_train_target[:,1].detach().cpu().numpy())
    ########

    start_time = t.perf_counter()
    steps=[]
    loss_values=[]
    global_steps = 0
    for epoch in range(epochs+1):     #it performs 1000 updates (10 in each epoch)
        x=[]
        y_sine=[]
        y_cose=[]
        y_tan=[]
        for x_batch,y_batch in loader:

            y_pred = model(x_batch)

            loss = loss_fn(y_pred,y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            

            ############
            x.extend(x_batch.detach().cpu().numpy())
            y_sine.extend(y_pred[:,0].detach().cpu().numpy())
            y_cose.extend(y_pred[:,1].detach().cpu().numpy())
            ############
        if epoch % 50 == 0:
            ##################
            sin_pred.set_data(x,y_sine)
            cos_pred.set_data(x,y_cose)

            steps.append(epoch)
            loss_values.append(loss.item())

            loss_gr.set_data(steps,loss_values)
            ax2.set_title(f"EPOCH : {epoch}/{epochs} \n LOSS : {loss.item()}")
            yield fig1,fig2,fig3
            ##################

            print(f"|{epoch:<30}|{round(loss.item(),4):<30}|")
    print("+------------------------------+------------------------------+")
    end_time = t.perf_counter()
    tot_time = end_time - start_time
    # print(f"\nexecution time with batching : {tot_time:.2f}")    


def main():
    x_train = torch.linspace(-torch.pi,torch.pi,1000).unsqueeze(1).to('cuda')
    # x_train = torch.randn(1000,1).to('cuda')
    y1=torch.sin(x_train)
    y2=torch.cos(x_train)
    
    y_train_target = torch.cat([y1,y2],dim=1)

    # x_test = torch.randn(50000,1).to('cuda')
    # y_test_target = torch.sin(x_test)

    model = nn.Sequential(
        nn.Linear(1,25), 
        nn.Tanh(), 
        nn.Linear(25,15),
        nn.Tanh(),
        nn.Linear(15,10),
        nn.GELU(),
        nn.Linear(10,2)
    )

    model.to('cuda')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.001
    )

    loss_fn = nn.MSELoss()


    # run_without_batching(model,optimizer,loss_fn,x_train,y_train_target)
    # print("\n")
    yield from run_with_batching(model,optimizer,loss_fn,x_train,y_train_target)




with gr.Blocks() as demo:
    gr.Markdown("TRIGONOMETRIC PREDICTION MODEL")
    with gr.Column():
        with gr.Row():
            sin_plot = gr.Plot()
            cos_plot = gr.Plot()
        loss_plot = gr.Plot()
    start_bttn = gr.Button('START')
    start_bttn.click(
        fn=main,
        inputs=None,
        outputs=[sin_plot,loss_plot,cos_plot]
    )

demo.launch()