import torch
import torch.nn as nn
from transformers import DistilBertModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from data_loader import train_df
from preprocessing import train_dataset, val_dataset, test_dataset
import copy
from sklearn.metrics import f1_score, roc_auc_score

# Define the model: DistilBERT + MLP (Multi-Layer Perceptron)
class DistilBERTMLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Define the MLP classifier on top of DistilBERT's output
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),             # GELU activation function
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(64, 2)
        )

        # Initialize weights of the MLP
        self._init_weights()

    # Function to initialize the MLP weights
    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                # Xavier normal initialization for weights
                nn.init.xavier_normal_(layer.weight)
                # Initialize bias to 0
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    # Forward pass
    def forward(self, input_ids, attention_mask):
        # Pass through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling over all tokens
        mask = attention_mask.unsqueeze(-1)
        masked_hidden = last_hidden_state * mask
        pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1)

        # Pass the pooled output through the MLP for classification
        return self.mlp(pooled)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Configuration training
BATCH_SIZE = 16
EPOCHS = 6
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 100

# Compute class weights based on label distribution
class_counts = train_df['label'].value_counts().sort_index() # Count the number of samples per class
# Inverse frequency == less represented classes receive higher weights
class_weights = torch.tensor((1. / class_counts).values, dtype=torch.float32).to(device)

# Dataloaders
from torch.utils.data import RandomSampler, SequentialSampler

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE
)

val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=BATCH_SIZE
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=BATCH_SIZE
)


# Inizializzazione modello
model = DistilBERTMLPClassifier().to(device)

# Gradual unfreezing strategy
def set_requires_grad(model, epoch):
    # Phase 0: only train the MLP classifier
    if epoch == 0:
        for param in model.distilbert.parameters():
            param.requires_grad = False   # Freeze all DistilBERT parameters
        for param in model.mlp.parameters():
            param.requires_grad = True    # Train only the MLP head

    # Phase 1: unfreeze layer 5
    elif epoch == 1:
        for param in model.distilbert.transformer.layer[5].parameters():
            param.requires_grad = True    # Unfreeze the last transformer layer
        for param in model.distilbert.transformer.layer[:5].parameters():
            param.requires_grad = False   # Keep the other layers frozen
        for param in model.mlp.parameters():
            param.requires_grad = True    # Keep training the MLP head

    # Phase 2: unfreeze layer 4
    elif epoch == 2:
        for param in model.distilbert.transformer.layer[4].parameters():
            param.requires_grad = True    # Unfreeze the second-to-last transformer layer
        for param in model.distilbert.transformer.layer[:4].parameters():
            param.requires_grad = False   # Keep lower layers frozen (0–3)
        for param in model.mlp.parameters():
            param.requires_grad = True

    # Phase 3: unfreeze layer 3
    elif epoch == 3:
        for param in model.distilbert.transformer.layer[3].parameters():
            param.requires_grad = True    # Unfreeze layer 3
        for param in model.distilbert.transformer.layer[:3].parameters():
            param.requires_grad = False   # Keep the first 3 layers frozen
        for param in model.mlp.parameters():
            param.requires_grad = True

    # Phase 4+: unfreeze everything
    else:
        for param in model.parameters():
            param.requires_grad = True    # Train the entire model


# Optimizer with layer-wise learning rates
optimizer = AdamW([
    # Layers 0-1 of DistilBERT: very small learning rate for stable fine-tuning
    {'params': [p for n, p in model.distilbert.named_parameters() if "layer.0" in n or "layer.1" in n], 'lr': 1e-5},
    # Layers 2-3: slightly higher learning rate
    {'params': [p for n, p in model.distilbert.named_parameters() if "layer.2" in n or "layer.3" in n], 'lr': 1.5e-5},
    # Layers 4-5: higher learning rate, closer to task-specific adaptation
    {'params': [p for n, p in model.distilbert.named_parameters() if "layer.4" in n or "layer.5" in n], 'lr': 2e-5},
    # MLP classifier: highest learning rate as it is trained from scratch
    {'params': model.mlp.parameters(), 'lr': 2e-5}
], weight_decay=WEIGHT_DECAY, eps=1e-8)

# Cosine annealing scheduler with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1)

# Weighted cross-entropy loss
criterion = nn.CrossEntropyLoss(weight=class_weights)


# 9. Training loop
def train_model():
    best_val_accuracy = 0
    best_model_state = None
    early_stop_counter = 0
    PATIENCE = 3
    training_stats = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)
        set_requires_grad(model, epoch)
        model.train()
        total_train_loss, total_train_correct = 0, 0

        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            model.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + step / len(train_dataloader))

            total_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_train_correct += (preds == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = total_train_correct / len(train_dataset)

        # Validation
        model.eval()
        total_val_loss, total_val_correct = 0, 0
        all_preds, all_labels = [], []

        for batch in val_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = total_val_correct / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds)
        val_roc_auc = roc_auc_score(all_labels, all_preds)

        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy,
            'val_f1': val_f1,
            'val_roc_auc': val_roc_auc
        })

        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)
    return model, training_stats

trained_model, stats = train_model()

