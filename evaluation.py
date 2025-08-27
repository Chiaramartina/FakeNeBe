import torch
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from training import trained_model, test_dataloader, device

def evaluate_metrics(model, dataloader, device):
    model.eval()
    total_correct = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids, attention_mask)

            # Probabilità per la classe positiva
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = total_correct / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, digits=4)

    print(f"Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}, Test ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    return accuracy, f1, roc_auc, all_labels, all_probs

test_accuracy, test_f1, test_roc_auc, all_labels, all_probs = evaluate_metrics(trained_model, test_dataloader, device)


def plot_roc_curve(all_labels, all_probs, roc_auc):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

plot_roc_curve(all_labels, all_probs, test_roc_auc)
