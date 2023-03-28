# Developer: Cnino
# This module it was create to loading the model
import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import warnings
import torch.nn.functional as F


def make_prediction(model, loader, output_csv="submission.csv"):
    preds = []
    filenames = []
    model.eval()

    for x, y, files in tqdm(loader):
        # print("files", files)
        x = x.to(config.DEVICE)
        with torch.no_grad():
            #pred = model(x).argmax(1) # Para el caso de pre procesamiento con nn.CrossEntropyLoss()
            predictions = model(x) # Para el caso de MSELoss
            # print("----------predictions", predictions )
            #print("pred", pred)
            #print("pred.cpu().numpy", pred.cpu().numpy)
            # Convert MSE floats to integer predictions
            predictions[predictions < 0.79] = 0 # Para el caso de procesamiento 2
            predictions[(predictions >= 0.79) & (predictions <= 1.35)] = 1 # Para el caso de MSELoss
            predictions[(predictions >= 1.35) & (predictions <= 100)] = 2 # Para el caso de procesamiento 2
            predictions = predictions.long().squeeze(1) # Para el caso de MSELoss
            preds.append(predictions.cpu().numpy()) # Para el caso deMSELoss
            #preds.append(t.cpu().numpy) # Para el caso de procesamiento nn.CrossEntropyLoss()
            filenames += files 
    # print("Valor de pred", predictions, "preds", preds, "pred.shape", predictions.shape)
    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    #model.train()
    print("\n/////////////// ************** Done with predictions ************** ////////////////")


def check_accuracy(loader, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    for x, y, filename in tqdm(loader):
        # print("filename", filename)
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            #scores = model(x) # para el caso de pre proceso nn.CrossEntropyLoss()
            predictions = model(x) # Para el caso de procesamiento MSELoss

        #_, predictions = scores.max(1) # para el caso de pre proceso nn.CrossEntropyLoss()
        #num_correct += (predictions == y).sum() # para el caso de pre proceso nn.CrossEntropyLoss()
        #num_samples += predictions.shape[0] # para el caso de pre proceso nn.CrossEntropyLoss()
        # print("/////////////**********predictions", predictions )
        #Convert MSE floats to integer predictions
        predictions[predictions < 0.79] = 0 # Para el caso de procesamiento 2
        predictions[(predictions >= 0.79) & (predictions <= 1.35)] = 1 # Para el caso de MSELoss
        predictions[(predictions >= 1.35) & (predictions <= 100)] = 2 # Para el caso de procesamiento 2
        predictions = predictions.long().view(-1) # Para el caso de procesamiento 2
        y = y.view(-1) # Para el caso de procesamiento 2

        num_correct += (predictions == y).sum() # Para el caso de procesamiento 2
        num_samples += predictions.shape[0] # Para el caso de procesamiento 2

        # add to lists
        all_preds.append(predictions.detach().cpu().numpy()) # Para el caso de procesamiento 1
        all_labels.append(y.detach().cpu().numpy()) # Para el caso de procesamiento 1
        
    print(
        f"------------ Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )
    model.train()
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(
        all_labels, axis=0, dtype=np.int64
    )


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("\n=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("\n=======> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_csv_for_blend(loader, model, output_csv_file):
    # warnings.warn("Important to have shuffle=False (and to ensure batch size is even size) when running get_csv_for_blend also set val_transforms to train_loader!")
    model.eval()
    filename_first = []
    filename_second = []
    labels_first = []
    labels_second = []
    all_features = []

    for idx, (images, y, image_files) in enumerate(tqdm(loader)):
        images = images.to(config.DEVICE)

        with torch.no_grad():
            features = F.adaptive_avg_pool2d(
                model.extract_features(images), output_size=1
            )
            features_logits = features.reshape(features.shape[0] // 2, 2, features.shape[1])
            preds = model(images).reshape(images.shape[0] // 2, 2, 1)
            new_features = (
                torch.cat([features_logits, preds], dim=2)
                .view(preds.shape[0], -1)
                .cpu()
                .numpy()
            )
            all_features.append(new_features)
            filename_first += image_files[::2]
            filename_second += image_files[1::2]
            labels_first.append(y[::2].cpu().numpy())
            labels_second.append(y[1::2].cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    df = pd.DataFrame(
        data=all_features, columns=[f"f_{idx}" for idx in range(all_features.shape[1])]
    )
    df["label_first"] = np.concatenate(labels_first, axis=0)
    df["label_second"] = np.concatenate(labels_second, axis=0)
    df["file_first"] = filename_first
    df["file_second"] = filename_second
    df.to_csv(output_csv_file, index=False)
    model.train()
