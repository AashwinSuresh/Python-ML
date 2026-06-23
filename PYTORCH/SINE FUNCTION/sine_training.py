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
    sine_real,=ax1.plot([],[],linestyle='--',label='Sine Graph')         
    sine_pred,=ax1.plot([],[],color='red',label='Predicted Sine curve')
    loss_gr, = ax2.plot([],[],color='blue',label="Loss Calculated")
    ax1.legend()
    ax2.legend()

    ax1.set_xlim(x_train.min().item(),x_train.max().item())
    ax1.set_ylim(-1,1)

    ax2.set_xlim(0,epochs+2)
    ax2.set_ylim(0,0.7)
    sine_real.set_data(x_train.detach().cpu().numpy(),y_train_target.detach().cpu().numpy())
    ########
    start_time = t.perf_counter()
    steps=[]
    loss_values=[]
    global_steps = 0
    for epoch in range(1,epochs+1):     #it performs 1000 updates (10 in each epoch)
        x=[]
        y=[]
        for x_batch,y_batch in loader:

            y_pred = model(x_batch)

            loss = loss_fn(y_pred,y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            

            ############
            x.extend(x_batch.detach().cpu().numpy())
            y.extend(y_pred.detach().cpu().numpy())
            ############
        if epoch % 100 == 0:
            ##################
            sine_pred.set_data(x,y)
            steps.append(global_steps)
            loss_values.append(loss.item())
            loss_gr.set_data(steps,loss_values)
            global_steps+=10
            ax2.set_title(f"EPOCH : {epoch}/{epochs} \n LOSS : {loss.item()}")
            yield fig1,fig2
            ##################

            print(f"|{epoch:<30}|{round(loss.item(),4):<30}|")
    print("+------------------------------+------------------------------+")
    end_time = t.perf_counter()
    tot_time = end_time - start_time
    # print(f"\nexecution time with batching : {tot_time:.2f}")    


def main():
    x_train = torch.linspace(-1.4,1.4,10000).unsqueeze(1)
    y_train_target = torch.sin(x_train)

    # x_test = torch.randn(50000,1).to('cuda')
    # y_test_target = torch.sin(x_test)

    model = nn.Sequential(
        nn.Linear(1,25), 
        nn.ReLU(), 
        nn.Linear(25,1) 
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
    with gr.Row():
        sin_plot = gr.Plot()
        loss_plot = gr.Plot()
    start_bttn = gr.Button('START')
    start_bttn.click(
        fn=main,
        inputs=None,
        outputs=[sin_plot,loss_plot]
    )

demo.launch()