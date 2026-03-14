import torch
from dataloader import create_dataset,create_dataloader
from data_visuliazation import visulize
from model import model
from train import train
from utils import save_model

train_data , test_data = create_dataset()
train_dataloader , test_dataloader = create_dataloader(train_data,test_data)
#visulize(train_data)

model = model(1,10,10)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

results = train(
    model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    5
)

save_model(
    model,
    "models",
    "model.pth"
)