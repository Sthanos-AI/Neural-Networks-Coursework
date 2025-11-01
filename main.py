import pandas as pd
from setup_environment import set_seed, check_device
from preprocess_dataset import load_and_preprocess
from model_process.model_factory import get_model
from model_process.model_train import train_model
from model_process.model_evaluate import evaluate_model

set_seed(42)
device = check_device()
tokenized_dataset, tokenizer = load_and_preprocess()

models = ["sshleifer/tiny-gpt2", "distilgpt2", "gpt2"]

results = []

for model_name in models:
    print(f"Training {model_name}...")
    model = get_model(model_name, tokenizer=tokenizer)
    trainer, metrics = train_model(model, tokenizer, tokenized_dataset, model_name)
    tokenized_test = tokenized_dataset["test"].shuffle(seed=42).select(range(3040))
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    labels, preds, acc, f1, loss = evaluate_model(trainer, tokenized_test)
    results.append({
        "Model": model_name,
        "Accuracy": acc["accuracy"],
        "Macro_F1": f1,
        "Eval_Loss": loss
    })

df_results = pd.DataFrame(results)
print("\nFinal Results:")
print(df_results)
