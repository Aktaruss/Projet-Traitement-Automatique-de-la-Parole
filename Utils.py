import pickle
from torch.utils import data
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

def get_basic_dataset(filename):
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    return dataset

class SpeechCommandDataset(data.Dataset):
    def __init__(self, signals, labels, metadata, transform_type=None):
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                "n_fft": 512,
                "n_mels": 40, 
                "win_length": int(16000 * 0.03),
                "hop_length": int(16000 * 0.01),
                "center": False 
            }
        )
        signals = [torch.tensor(sig, dtype=torch.float32) for sig in signals]

        processed_signals = []

        if transform_type == "MFCC":
            for sig in signals:
                if sig.ndim == 1:
                    sig = sig.unsqueeze(0)
                mfcc = self.transform(sig)
                processed_signals.append(mfcc)

        self.signals = torch.stack(processed_signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)

def model_summary(model):
    total = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total += param
        print(f"Couche({name}) : {param} (Paramètres)")
    print(f"Total : {total} (Paramètres)")

def evaluate(model, loader, device):
    criterion=nn.CrossEntropyLoss()
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss += criterion(logits,labels).numpy()
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, prediction = torch.max(logits, 1)
            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    correct = sum([all_preds[i] == all_labels[i] for i in range(len(all_preds))])
    accuracy = 100 * correct / len(all_labels)
    return accuracy, all_preds, all_labels, all_probs, loss/len(loader)

def train(model, train_loader, validation_loader, nb_steps=33000, val_step=400):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    lr_drop = nb_steps * (5/6)
    
    model.to(device)
    model.train()
    train_loss = []
    val_acc = []
    val_loss = []
    step = 0

    while step < nb_steps:
        for inputs, labels in train_loader:
            if step >= nb_steps:
                break

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.numpy())
        step += 1

        if step == lr_drop:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        
        if step % val_step == 0:
            acc, _, _, _, loss = evaluate(model,validation_loader,device)
            val_acc.append(acc)
            val_loss.append(loss)

    return model, train_loss, val_acc, val_loss