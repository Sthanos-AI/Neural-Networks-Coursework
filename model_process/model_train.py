from transformers import Trainer, TrainingArguments

def train_model(model, tokenizer, tokenized_dataset, model_name):
    train_ds = tokenized_dataset["train"]
    test_ds = tokenized_dataset["test"]

    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,
        report_to="none",
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    metrics = trainer.evaluate()
    return trainer, metrics
