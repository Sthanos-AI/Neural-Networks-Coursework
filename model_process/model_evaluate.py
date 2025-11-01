import evaluate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(trainer, tokenized_test):
    preds_output = trainer.predict(tokenized_test)

    preds = np.argmax(preds_output.predictions, axis=1)
    refs = np.array(tokenized_test["labels"])  # convert tensor to numpy array

    accuracy = evaluate.load("accuracy").compute(predictions=preds, references=refs)
    f1 = evaluate.load("f1").compute(predictions=preds, references=refs, average="macro")

    # Confusion matrix
    cm = confusion_matrix(refs, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["World","Sports","Business","Sci/Tech"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return refs, preds, accuracy, f1, preds_output.metrics.get("eval_loss", None)
