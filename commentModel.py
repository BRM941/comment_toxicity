import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class Data(Dataset):
    def __init__(self, df, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.X = encoded["input_ids"]
        self.Y = labels
        self.len = df.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index1):
        sample = self.X[index1].float(), torch.tensor(self.Y[index1], dtype=torch.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.layer1 = nn.GRU(input_size=469, hidden_size=469)
        self.layer2 = nn.GRU(input_size=469, hidden_size=469)
        self.layer3 = nn.Linear(in_features=469, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x[0])
        x = self.layer2(x)
        x = self.relu(x[0])
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

if(__name__ == "__main__"):
    torch.manual_seed(0)

    try:
        df = pd.read_csv("./toxicity_en.csv")
    except Exception as e:
        print(e)


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = ":"

    text = df['text'].to_list()
    encoded = tokenizer(text, padding=True, return_tensors='pt')

    toxicity = df['is_toxic']
    encoder = LabelEncoder()
    labels = encoder.fit_transform(toxicity)

    data = Data(df)
    train_set, test_set = torch.utils.data.random_split(data, [0.8, 0.2])
    train_set, val_set = torch.utils.data.random_split(data, [0.8, 0.2])

    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_set)
    val_loader = DataLoader(dataset=val_set)

    model = Model()
    BCE = nn.BCELoss()
    op = optim.Adam(model.parameters(), lr=0.01)

    def validation(model, valid_loader, loss_function):
        model.eval()
        loss_total = 0

        with torch.no_grad():
            for x_val, y_val in valid_loader:
                predictions = model(x_val)
                val_BCE = loss_function(torch.flatten(predictions), y_val)
                loss_total += val_BCE.item()
        return loss_total / len(valid_loader)

    training_loss = []
    validation_loss = []
    epochs = []

    num_epochs = 100
    for i in range(num_epochs):
        for x,y in train_loader:
            prediction = model(x)
            loss = BCE(torch.flatten(prediction), y)
            loss.backward()
            op.step()
            op.zero_grad()

        val_loss = validation(model, val_loader, BCE)
        training_loss.append(loss.item())
        validation_loss.append(val_loss)
        epochs.append(i)
        
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{i + 1}/{num_epochs}], BCE Loss: {loss.item()}')
            torch.save(model.state_dict(), "model.pth")
