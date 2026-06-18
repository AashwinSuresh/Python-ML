import torch
import numpy as np
import gradio as gr
import pandas as pd 
from sklearn.metrics import (accuracy_score,precision_score,f1_score,recall_score)
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments




def compute_metrics(eval_pred):
    logits,label = eval_pred
    predictions = np.argmax(logits,axis=1)
    accuracy = accuracy_score(label,predictions)
    precision = precision_score(label,predictions)
    recall = recall_score(label,predictions)
    f1 = f1_score(label,predictions)
    return {
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1
            }
    
def evaluate(tokenized_dataset,model):
       
    training_args = TrainingArguments(
        per_device_eval_batch_size=32,
        fp16=True,
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics
    )

    evaluation = trainer.evaluate()
    print("\nTHE EVALUATIONS ARE : ")
    print("---------------------\n")
    print(evaluation)
    loss=evaluation['eval_loss']
    acc=evaluation['eval_accuracy']
    pre=evaluation['eval_precision']
    rec=evaluation['eval_recall']
    f1=evaluation['eval_f1_score']

    return loss,acc,pre,rec,f1


def predict(tokenized_dataset,model):
    labels = tokenized_dataset["labels"]
    training_args = TrainingArguments(
        fp16=True,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
    )

    predictions = trainer.predict(tokenized_dataset)
    predictions = predictions.predictions
    predictions = np.argmax(predictions,axis=1)
    acc = accuracy_score(labels,predictions)
    print(f"ACCURACY IS : {acc}")
    return acc

def main(file_obj):
    if file_obj is None:
        return " Please upload a valid file "
    file_path = file_obj.name
    df_test = pd.read_csv(file_path)
    df_test['labels'] = df_test['labels'].replace("neg",0).replace("pos",1)
    print(df_test.head())

    
    dataset = Dataset.from_pandas(df_test)
    tokenizer = AutoTokenizer.from_pretrained(r"..\models\BERT")
    model = AutoModelForSequenceClassification.from_pretrained(r"..\models\BERT")

    
    if torch.cuda.is_available():
        print("\n\n####################")
        print("## RUNNING IN GPU ##")
        print("####################\n\n")

        device = torch.device("cuda")
    else :
        print("\n\n####################")
        print("## RUNNING IN CPU ##")
        print("####################\n\n")
        device = torch.device("cpu")
    
    model.to(device)

    # ########################################
    # ##  SAMPLE TEST  WITH ONLY ONE INPUT  ##
    # ########################################
    
    # sample_test = "I am not liking the plot of this movie very much"
    # encoding = tokenizer(
    #     sample_test,
    #     truncation = True,
    #     padding = True,
    #     max_length = 256,
    #     return_tensors = "pt"       # during manual (without using trainer()) we need to convert it to tensors for the model to understand it 
    # )                               # but when using trainer() a special function inside it will automatically carry out the conversions
    
    # encoding.to(device)
    # prediction = model(**encoding)

    # print(f"type of prediction : {type(prediction)} : \nprediction value : {prediction.logits}")
    # logits = torch.argmax(prediction.logits,dim=1)
    # print(f"sub type : {type(logits)}  \nvalue : {logits} , prediction : {logits.item()}")
    
    tokenized_dataset = dataset.map(
        lambda x : tokenizer(
            x["text"],
            padding = True,
            truncation = True,
            max_length = 256,
        ),
        batched = True
    )

    return evaluate(tokenized_dataset,model)
    #return predict(tokenized_dataset,model)

  

    


if __name__ == "__main__":
    iface = gr.Interface(
        fn = main,
        inputs = gr.File(file_types = ['.csv'],label="UPLOAD DOCUMENT FOR PREDICTION"),
        outputs = [
            gr.Textbox(label="EVALUATION LOSS : "),
            gr.Textbox(label="ACCURACY :"),
            gr.Textbox(label="PRECISION : "),
            gr.Textbox(label="RECALL : "),
            gr.Textbox(label="F1 SCORE : ")
        ]
    )
    
    iface.launch()

