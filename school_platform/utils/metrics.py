import torch
import torch.nn as nn


@torch.no_grad()
def evaluate_model(model, data_loader, device="cpu"):
    """Evaluate model on a data loader. Returns (loss, accuracy).

    Supports both 3-tuple (input_ids, attention_mask, labels) for DistilBERT
    and 2-tuple (features, labels) for MLP classifiers.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in data_loader:
        if len(batch) == 3:
            # Tokenized: (input_ids, attention_mask, labels)
            input_ids, attn_mask, y = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attn_mask)
        else:
            # Embedding: (X, y)
            X, y = batch[0].to(device), batch[1].to(device)
            outputs = model(X)
        loss = criterion(outputs, y)
        total_loss += loss.item() * y.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
