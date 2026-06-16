import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments



print(torch.cuda.is_available())


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2
)

df_train = pd.read_csv(r"datasets\imdb\train.csv")
df_train['labels'] = df_train['labels'].replace('neg', 0).replace('pos',1)
print(df_train.head())

train_dataset = Dataset.from_pandas(df_train)

tokenized_train_dataset = train_dataset.map(
    lambda x:tokenizer(
        x['text'],
        padding =True,
        truncation = True
    ),
    batched = True
)



training_args = TrainingArguments(
    output_dir = r".\results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=tokenized_train_dataset
)

print("currently using : ",next(model.parameters()).device)
trainer.train()

