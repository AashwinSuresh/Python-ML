import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding

def main():
    print(torch.cuda.is_available())


    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels = 2
    )

    df_train = pd.read_csv(r"..\datasets\bert_movie_review\train.csv")
    df_train['labels'] = df_train['labels'].replace('neg', 0).replace('pos',1)
    print(df_train.head())

    train_dataset = Dataset.from_pandas(df_train)

    tokenized_train_dataset = train_dataset.map(
    lambda x:tokenizer(
        x['text'],
        truncation = True,
        padding = True,
        max_length = 256
    ),
    batched = True
    )

    # # Dynamic padding
    # data_collator = DataCollatorWithPadding(        
    # tokenizer=tokenizer 
    # )


    training_args = TrainingArguments(
    output_dir = r".\results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=True,
    dataloader_num_workers = 0,     #CPU uses 4 separate processes to prepare batches in parallel to feed it to GPU.
                                    #dataloader makes the process much faster which avoids gpu waiting while cpu fetches batches and convert it to tensors
    )                                   
    trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=tokenized_train_dataset,
    )

    print("currently using : ",next(model.parameters()).device)
    trainer.train()
    trainer.save_model(r"..\models\BERT")
    tokenizer.save_pretrained(r"..\models\BERT")

if __name__ == "__main__":
    main()
