import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class Data(Dataset):
    def __init__(self, df, tokenizer) -> None:
        super().__init__()
        self.X = df['comment_text']
        self.Y = df["toxic"]
        self.len = df.shape[0]
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len
    
    def __getitem__(self, index1):
        encoded = self.tokenizer.encode_plus(self.X[index1], padding='max_length', truncation=True, max_length=100, add_special_tokens=True)
        sample = torch.tensor(encoded['input_ids'], dtype=torch.int64), torch.tensor(self.Y[index1], dtype=torch.int64)
        return sample

class Model(nn.Module):
    def __init__(self, pad_id) -> None:
        super().__init__()

        self.embed = nn.Embedding(50258, 50, padding_idx=pad_id)
        self.layer1 = nn.GRU(input_size=50, hidden_size=200, num_layers=2, batch_first=True, bidirectional=True, dropout=0)
        self.layer2 = nn.Linear(in_features=400, out_features=256)
        self.dropout = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(in_features=256, out_features=350)
        self.layer4 = nn.Linear(in_features=350, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x, _= self.layer1(x)
        x =x[:, -1, :]
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        return x



if(__name__ == "__main__"):
    torch.manual_seed(0)

    try:
        df = pd.read_csv("./train.csv")
    except Exception as e:
        print(e)


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', pad_token="<pad>", unk_token="<pad>")

    def get_training_corpus():
        return (
            df["comment_text"][i : i + 1000]
            for i in range(0, len(df["comment_text"]), 1000)
        ) 
    
    training_corpus = get_training_corpus()
    tokinzer = tokenizer.train_new_from_iterator(training_corpus, 500257)
    tokenizer.save_pretrained("comment_text")
    
    data = Data(df, tokenizer)

    train_set, val_set = torch.utils.data.random_split(data, [0.8, 0.2])
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=True, num_workers=8, prefetch_factor=3)
    val_loader = DataLoader(dataset=val_set, shuffle=True)

    model = Model(pad_id=tokenizer.pad_token_id)
    model = torch.jit.script(model)
    BCE = nn.BCELoss()
    op = optim.Adam(model.parameters(), lr=0.001)

    def validation(model, valid_loader, loss_function):
        model.eval()
        loss_total = 0

        with torch.no_grad():
            for x_val, y_val in valid_loader:
                predictions = model(x_val)
                val_loss = loss_function(predictions.float(), y_val.float())
                loss_total += val_loss
        return loss_total / len(valid_loader)
    
    
    num_epochs = 4
    for i in range(num_epochs):
        for j, (x,y) in enumerate(train_loader):
            op.zero_grad()
            prediction = model(x)
            prediction = torch.squeeze(prediction)
            loss = BCE(prediction.float(), y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            op.step()
            

            if (j + 1) % 100 == 0:
                print(f'Epoch [{i + 1}/{num_epochs}], Step: {j + 1}/{len(train_loader)},  BCE Loss: {loss.item():.4f}')
                torch.save(model.state_dict(), "model.pth")

        
        
        
        if (i + 1) % 2 == 0:
            print(f'Epoch [{i + 1}/{num_epochs}],  BCE Loss: {loss.item():.4f}')
            torch.save(model.state_dict(), "model.pth")

        if (i + 1) == num_epochs:
            val_loss = validation(model, val_loader, BCE)
            print(f'Validation Loss: {val_loss:.4f}')
            pass
    
