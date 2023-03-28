# Developer: Cnino
# This modul was created to train the model and to load the data
from statistics import mode
import torch
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from torchvision.utils import save_image
import pandas as pd #For reading csv files.
import numpy as np 
import matplotlib.pyplot as plt #For plotting.
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
        # Get data to cuda if possible

        # save examples and make sure they look ok with the data augmentation,
        # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
        # save_image(data, f"hi_{batch_idx}.png")

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            #loss = loss_fn(scores, targets) # Caso nn.CrossEntropyLoss()
            loss = loss_fn(scores, targets.unsqueeze(1).float()) # Para el caso MSELoss

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

        #print(f"++++++++++++++++ Loss average over epoch: {sum(losses)/len(losses)}")

    print(f"\n++++++++++++++++ Loss average over epoch: {sum(losses)/len(losses)}")


def main():
    train_ds = DRDataset(
        # images_folder="./trainejem150/",
        images_folder="G:/Mi unidad/Colab Notebooks/retinopathy/DiaHiper/Train/total_650/",
        path_to_csv="./trainLabels.csv",
        # transform=config.val_transforms, # En el caso de generar el preprocesammiento 2
        transform=config.train_transforms, # Para el caso del procesamiento1
    )
    val_ds = DRDataset(
        images_folder="G:/Mi unidad/Colab Notebooks/retinopathy/DiaHiper/Train/total_650/",
        path_to_csv="./valLabels.csv",
        transform=config.val_transforms,
    )
    test_ds = DRDataset(
        images_folder="G:/Mi unidad/Colab Notebooks/retinopathy/DiaHiper/Test/total_650/",
        path_to_csv="./testLabels.csv",
        transform=config.val_transforms,
        train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True, # Para el entrenamiento 1
        # shuffle=False, # Para el entrenamiento 2
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    #Histogram of label counts.
    # path = "./"

    # train_df = pd.read_csv(f"{path}trainLabels.csv")
    # print(f'No.of.training_samples: {len(train_df)}')
    # print(train_df)
    # print("df:", train_df)
    # train_df.level.hist()
    # plt.xticks([0,1,2,3,4])
    # plt.grid(False)
    # plt.show() 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Use GPU if it's available or else use CPU.
    # #As you can see,the data is imbalanced.
    # #So we've to calculate weights for each class,which can be used in calculating loss.
    # from sklearn.utils import class_weight #For calculating weights for each class.
    # class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y=train_df['level'].values)
    # class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    
    # loss_fn =  nn.CrossEntropyLoss(weight=class_weights)
    #loss_fn =  nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, 1)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    #model._fc = nn.Linear(1536, 1)

    # Run after training is done and you've achieved good result
    # on validation set, then run train_blend.py file to use information
    # about both eyes concatenated
    # get_csv_for_blend(val_loader, model, "./train/val_blend.csv")
    # get_csv_for_blend(train_loader, model, "./train/train_blend.csv")
    # get_csv_for_blend(test_loader, model, "./train/test_blend.csv")
    # make_prediction(model, test_loader, "submission_.csv")

    # Generar las predicciones una a la vez para asegurar que funciona
    # make_prediction(model, test_loader)
    # import sys
    # sys.exit()
    
    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # get on validation
        preds, labels = check_accuracy(val_loader, model, config.DEVICE)
        print(f"\n---------------------- QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # save_checkpoint(checkpoint, filename=f"b3_{epoch}.pth.tar")
            save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

    make_prediction(model, test_loader, "submission_.csv")

if __name__ == "__main__":
    main()
