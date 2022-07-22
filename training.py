import torch
import torch.nn as nn


def train(model, train_dl, num_epochs):
    """ Training loop, printing accuracy"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    for epoch in range(num_epochs):
        correct_prediction = 0
        running_loss = 0
        total_prediction = 0
        for i, data in enumerate(train_dl):
            inputs, labels = data[0], data[1].reshape(-1, 1)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            prediction = ((outputs > 0.5) == labels).float()
            correct_prediction += prediction.sum()
            running_loss += loss.item() * inputs.size(0)
            total_prediction += prediction.shape[0]

            if i % 1000 == 0:
                print(f'[{epoch + 1}, {i + 1}] avg_loss: {running_loss / total_prediction :.2f}'
                      f' avg_accuracy: {correct_prediction / total_prediction:.2f} '
                      f'accuracy = {prediction.sum() / prediction.shape[0]:.2f}')

        accuracy = 100 * correct_prediction / total_prediction
        print(f"Accuracy = {accuracy}")

    print('Finished Training')


def test_model(dev_dl, model):
    """ Model testing, given dev/test data and model returns accuracy """
    correct_prediction = 0
    total_prediction = 0
    for i, data in enumerate(dev_dl):
        inputs, labels = data[0], data[1].reshape(-1, 1)
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        outputs = model(inputs)
        prediction = ((outputs > 0.5) == labels).float()
        correct_prediction += prediction.sum()
        total_prediction += prediction.shape[0]
    accuracy = 100 * correct_prediction / total_prediction
    return accuracy
