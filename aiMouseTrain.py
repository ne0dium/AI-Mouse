import numpy as np
import torch

from model import LSTM1, NeuralNet


def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        _y = data[(i + seq_length):(i + seq_length) + 1]
        x.append(_x)
        y.append(_y)

    return x, y


def train(pointsList, num_epochs, learning_rate, input_size, hidden_size, num_layers, num_classes, seq, WIDTH, HEIGHT,
          model_type="lstm"):
    newPointsList = []
    for pts in pointsList:
        newpts = []

        for i in range(0, len(pts) - 1):
            print(pts)
            newpts.append([pts[i][0], pts[i + 1][2]])
        newPointsList.append(newpts)

    allData = []

    for points in newPointsList:
        data = np.array(points)
        data = data / np.array([WIDTH, HEIGHT]).reshape(1, 1, -1)
        data = data.reshape(data.shape[0], -1)
        test = data.tolist()
        allData.append(test)

    # x, y = sliding_windows(test, 1)

    x = []
    y = []

    for item in allData[:-1]:
        x_, y_ = sliding_windows(item, seq)
        x.extend(x_)
        y.extend(y_)

    x, y = np.array(x), np.array(y)

    x = x[..., :2]
    y = y[..., -2:]

    x_train, y_train = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    y_train = torch.squeeze(y_train)

    device = torch.device(0)
    # device = 'cpu'

    # lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    if model_type == "lstm":
        model = LSTM1(num_classes, input_size, hidden_size, num_layers, 1)
    else:
        model = NeuralNet(2, 2)

    x_train = torch.squeeze(x_train, 1)

    model = model.to(device)

    x_train = x_train.float().to(device)
    y_train = y_train.float().to(device)

    split = int(0.2 * len(x_train))

    x_test = x_train[-split:]
    y_test = y_train[-split:]

    x_train = x_train[:split]
    y_train = y_train[:split]

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        if model_type == 'lstm':
            outputs = model(x_train, device)
        else:
            outputs = model(x_train)
        # outputs = model(x_train)
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, y_train)

        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            with torch.no_grad():
                if model_type == 'lstm':
                    outputs = model(x_test, device)
                else:
                    outputs = model(x_test)
                # outputs = model(x_train)
                optimizer.zero_grad()

                # obtain the loss function
                loss = criterion(outputs, y_test)
                print("Test loss: %1.5f" % (loss.item()))

    model.eval()
    model.cpu()

    return model
