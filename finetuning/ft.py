import os
from transformers import TrainingArguments, Trainer

from datasets import Dataset

def fine_tune(model, tokenizer, config, dataset_dir):

    model_dir = os.path.join(dataset_dir, "fine_tuned_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tokenizer=tokenizer
    def preprocessing(dataset):
        inputs = dataset["input"]
        targets = dataset["target"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
    
        labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_set = Dataset.load_from_disk(os.path.join(dataset_dir, "train_set"))
    val_set = Dataset.load_from_disk(os.path.join(dataset_dir, "val_set"))

    #tokenize datasets
    tokenized_train_dataset = train_set.map(preprocessing, batched=True)
    tokenized_val_dataset = val_set.map(preprocessing, batched=True)

    #def training args
    training_args = TrainingArguments(
        output_dir=dataset_dir,
        eval_strategy="epoch",
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        num_train_epochs=config['n_epochs'],
        weight_decay=config['weight_decay'],
        save_total_limit=2,
        logging_steps=200,
        report_to="none", 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    

    
        