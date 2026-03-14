import torch
from evaluation import evaluate_model, get_metrics
import torch.nn as nn
import config


def train_one_epoch(model, loader, optimizer, device, run):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for word_ids, char_ids, tag_ids, lengths in loader:
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        tag_ids  = tag_ids.to(device)
        lengths = lengths.to(device)

        model.zero_grad()
        if run == "SM":
            loss = model.softmax_margin(word_ids, char_ids, tag_ids, lengths, config.COST_FACTOR)
        else:
            loss = model.neg_log_likelihood(word_ids, char_ids, tag_ids, lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)

def train(model, train_loader, val_loader, epochs, optimizer, device, idx_to_tag, run=None):
    best_f1 = 0.0
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        import time
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device, run)
        elapsed = time.time() - start
        print(f"Epoch time: {elapsed:.2f}s")

        preds, golds = evaluate_model(model, val_loader, device)
        _, _, f1 = get_metrics(preds, golds, idx_to_tag)
        print(f"Epoch {epoch+1}/{epochs} — loss: {loss:.4f}, F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{run}_best.pt")

    return best_f1
