import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from models import SimpleModel

label_to_int = {'stand': 0,
                'sit': 1,
                'lay': 2,
                'stand_to_sit': 3,
                'stand_to_lie': 4,
                'sit_to_stand': 5,
                'sit_to_lie': 6,
                'lie_to_stand': 7,
                'lie_to_sit': 8,
                'walk': 9,
                'down': 10,
                'up': 11}

int_to_label = {0: 'stand',
                1: 'sit',
                2: 'lay',
                3: 'stand_to_sit',
                4: 'stand_to_lay',
                5: 'sit_to_stand',
                6: 'sit_to_lay',
                7: 'lay_to_stand',
                8: 'lay_to_sit',
                9: 'walk',
                10: 'down',
                11: 'up'}


def load_train_data():
    train_data = pd.read_csv('train.csv')

    train_features = train_data.loc[:, ~train_data.columns.isin(['Id', 'Activity'])]
    train_labels = train_data['Activity']

    train_labels = train_labels.map(label_to_int)

    return torch.tensor(train_features.values).float(), torch.tensor(train_labels.values).long()


def load_test_data():
    test_data = pd.read_csv('test.csv')

    test_features = test_data.loc[:, ~test_data.columns.isin(['Id'])]
    test_ids = test_data['Id']

    return torch.tensor(test_features.values).float(), test_ids.values


def train(train_dataset, val_split=0.1, batch_size=32, num_epochs=10):
    model = SimpleModel()
    model.train()

    run_with_val = val_split > 0

    if run_with_val:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=int(len(train_dataset) * val_split))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    all_train_losses = []
    all_val_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            features, labels = data
            optimizer.zero_grad()

            outputs = model(features)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataset)
        all_train_losses.append(train_loss)

        if run_with_val:

            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for i, data in enumerate(val_loader, 0):
                    features, labels = data

                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_preds.extend(torch.flatten(predicted).tolist())
                    all_labels.extend(torch.flatten(labels).tolist())

                    val_loss += loss.item()

                print(confusion_matrix(all_labels, all_preds))

            val_loss /= len(val_dataset)
            val_accuracy = correct / total
            all_val_losses.append(val_loss)

        if run_with_val:
            print(f"{train_loss = :.3f}, {val_loss = :.3f}, {val_accuracy = :.3f}")
        else:
            print(f"{train_loss = :.3f}")

    plt.plot(range(num_epochs), all_train_losses, 'r')
    if run_with_val:
        plt.plot(range(num_epochs), all_val_losses, 'b')
    plt.show()

    return model


def predict(model, test_dataset, test_ids):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    os.makedirs('predictions', exist_ok=True)
    output_filename = f"predictions/predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    output_data = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            features, = data
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            id = test_ids[i]

            output_data.append([id, int_to_label[predicted[0].item()]])

    output_df = pd.DataFrame(output_data, columns=["Id", "Activity"])
    output_df.to_csv(output_filename, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


def main():
    train_features, train_labels = load_train_data()
    print(train_features.shape, train_labels.shape)

    train_dataset = TensorDataset(train_features, train_labels)

    trained_model = train(train_dataset, val_split=0.1, num_epochs=1)

    test_features, test_ids = load_test_data()
    test_dataset = TensorDataset(test_features)

    predict(trained_model, test_dataset, test_ids)


if __name__ == '__main__':
    main()
