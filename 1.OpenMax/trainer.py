import torch
from tqdm import tqdm
from pathlib import Path

class Trainer:
    """
    A simple train class for training CNN model
    """
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
            
        self.best_acc = 0
        self.best_loss = 1000

        self.destination_folder = Path("./models")
        self.destination_folder.mkdir(parents=True, exist_ok=True)


    def train_one_epoch(self):
        train_loss = 0.0
        correct_train = 0
        total_train_samples = len(self.train_loader.dataset)

        self.model.train()
        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct_train += predicted.eq(labels).sum().item()

        train_accuracy =  (correct_train / total_train_samples) * 100
        return train_loss, train_accuracy


    def validate(self):
        valid_loss = 0.0
        correct_valid = 0
        total_valid_samples = len(self.valid_loader.dataset)

        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).double()
                
                loss = self.loss_fn(outputs, labels)
                valid_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct_valid += predicted.eq(labels).sum().item()

        valid_accuracy = (correct_valid / total_valid_samples) * 100
        return valid_loss, valid_accuracy


    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def fit(self, n_epochs, save_name, scheduler=None):
        for epoch in range(n_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            valid_loss, valid_accuracy = self.validate()

            print(
                f"Epoch [{epoch + 1}/{n_epochs}], "
                f"Training Loss: {train_loss:.6f}, Training Accuracy: {train_accuracy:.2f}%, "
                f"Validation Loss: {valid_loss:.6f}, Validation Accuracy: {valid_accuracy:.2f}%"
            )

            if scheduler:
                scheduler.step()

            if valid_accuracy > self.best_acc:
                self.best_acc = valid_accuracy
                self.save_model(self.destination_folder / f"{save_name}_best_acc.pth")

            if valid_loss <= self.best_loss:
                self.best_loss = valid_loss
                self.save_model(self.destination_folder / f"{save_name}_best_loss.pth")
