"""
Barak Beilin & Amir Jevnisek

Homework Solution for the first exercise.

Since MNIST server was done during the solution of this exercise, we took
MNIST dataset in a csv format from here:
https://www.kaggle.com/oddrationale/mnist-in-csv

To properly run:
+ download and extract the csvs to a dataset/ folder,
+ create a results/ folder
+ install environment with the supplied environment.yml attached.
+ run the script from cmd: python main.py

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = 'results'
TRAIN_SET_NAME = 'mnist_train.csv'
TEST_SET_NAME = 'mnist_test.csv'

BATCH_SIZE = 64
IMAGE_SIZE = 28


class MNISTfromCSV(Dataset):
    def __init__(self, is_train):
        if is_train:
            self.dataset_path = os.path.join("dataset", TRAIN_SET_NAME)
        else:
            self.dataset_path = os.path.join("dataset", TEST_SET_NAME)
        dataframe = pd.read_csv(self.dataset_path)
        self.images = torch.from_numpy(np.reshape(dataframe.drop(
            columns='label').values,
                                 (dataframe.shape[0],
                                  IMAGE_SIZE * IMAGE_SIZE)) / 255.0)
        self.labels = dataframe['label'].values

    def __getitem__(self, index):
        sample_dict = {'image': self.images[index].float(),
                       'label': self.labels[index]}
        return sample_dict

    def __len__(self):
        return len(self.labels)


def print_dataset_sample(is_train):
    dataset = MNISTfromCSV(is_train=is_train)
    plt.figure(figsize=(15, 15))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(dataset[i]['image'].reshape(IMAGE_SIZE, IMAGE_SIZE))
        plt.title(dataset[i]['label'])
        plt.axis('off')
    # plt.subplots_adjust(wspace=0.0, hspace=0.1)
    dataset_name = 'train' if is_train else 'test'
    plt.suptitle('MNIST ' + dataset_name)
    plt.savefig(os.path.join(RESULTS_DIR, 'MNIST_' + dataset_name + ".png"))


class FullyConnectedModel(nn.Module):
    def __init__(self):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 10)

    def forward(self, x):
        x = self.fc1(x)
        return x


class AnotherModel(nn.Module):
    def __init__(self):
        super(AnotherModel, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Trainer:
    def __init__(self, model, epochs=10, learning_scheme='Adam'):
        self.model = model.to(device)
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        if learning_scheme == 'Adam':
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                        lr=1e-2,)
        self.train_dataset = MNISTfromCSV(is_train=True)
        self.test_dataset = MNISTfromCSV(is_train=False)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=8)
        self.test_dataloader = DataLoader(self.test_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=8)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.train_single_epoch()
            train_accuracy = self.evaluate_accuracy_on_dataset(
                self.train_dataloader)
            test_accuracy = self.evaluate_accuracy_on_dataset(
                self.test_dataloader)
            print(f"Epoch # {epoch} : Train accuracy: {train_accuracy:.3f}, "
                  f"Test accuracy: {test_accuracy:.3f}")

    def evaluate_accuracy_on_dataset(self, dataloader):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch['image'].to(device), batch[
                    'label'].to(device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def train_single_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            inputs, labels = batch['image'].to(device), batch['label'].to(
                device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()


def get_correctly_classified_image(model, seed=999):
    torch.manual_seed(seed)
    test_dataset = MNISTfromCSV(is_train=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=8)
    model = model.to(device)
    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch['image'].to(device), batch[
                'label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            classification_result = predicted == labels
            if torch.any(classification_result):
                index_of_misclassified_image = classification_result.nonzero()[
                    0]
                image = images[index_of_misclassified_image]
                true_label = labels[index_of_misclassified_image].item()
                return image, true_label
    return None, None


def check_if_pixel_can_be_adversarial(model, image, true_label,
                                      pixel_to_change):
    pixel_range = torch.range(start=0.001, end=10, step=0.001)
    image_tensor = torch.cat([image] * len(pixel_range))
    image_tensor[..., pixel_to_change] = pixel_range
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predictions = torch.max(outputs.data, 1)
    return torch.any(predictions != true_label).item()


def get_pixel_new_value_and_label(model, image, true_label, pixel_to_change):
    pixel_range = torch.range(start=0.001, end=10, step=0.001)
    image_tensor = torch.cat([image] * len(pixel_range))
    image_tensor[..., pixel_to_change] = pixel_range
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predictions = torch.max(outputs.data, 1)
    different_prediction_indices = (predictions != true_label).nonzero()
    if different_prediction_indices.tolist():
        new_value = pixel_range[different_prediction_indices[0]].item()
        new_label = predictions[different_prediction_indices[0]].item()
    else:
        new_value, new_label = None, None
    return new_value, new_label


def create_adversarial_example(model, image, true_label, seed=999):
    model.eval()
    torch.manual_seed(seed)
    for _ in range(IMAGE_SIZE * IMAGE_SIZE):
        pixel_to_change = int(torch.randint(IMAGE_SIZE * IMAGE_SIZE, (1,)))
        if check_if_pixel_can_be_adversarial(model, image, true_label,
                                             pixel_to_change):
            break
    new_value, new_label = get_pixel_new_value_and_label(model,
                                                         image,
                                                         true_label,
                                                         pixel_to_change)
    adversarial_image = image.clone()
    adversarial_image[..., pixel_to_change] = new_value
    return adversarial_image, pixel_to_change, new_value, new_label


def log_and_get_adversarial_example(model, image, label, seed):
    adversarial_example, manipulated_pixel_index, new_value, new_label \
        = create_adversarial_example(model, image, label, seed)
    previous_value = image[..., manipulated_pixel_index].item()
    print(f"Pixel changed: {manipulated_pixel_index}, from "
          f"{previous_value:.2f} to {new_value:.3f}, and label from {label} to"
          f" {new_label}")
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE))
    axes[1].imshow(adversarial_example.reshape(IMAGE_SIZE, IMAGE_SIZE))
    axes[0].set_title('Original image')
    axes[1].set_title('Manipulated image')
    row, col = manipulated_pixel_index // IMAGE_SIZE, manipulated_pixel_index \
               % IMAGE_SIZE
    fig.suptitle(f'Image manipulated at ({row}, {col}) from '
                 f'{image[..., manipulated_pixel_index].item():.1f} '
                 f'to {new_value:.1f}; label: {label}->{new_label}')
    plt.savefig(os.path.join(RESULTS_DIR,
                             f'adversarial_example_seed_{seed}.png'))
    return adversarial_example, manipulated_pixel_index, new_value, new_label


def main():
    print_dataset_sample(is_train=True)
    print_dataset_sample(is_train=False)
    model = FullyConnectedModel()
    fc_trainer = Trainer(model, epochs=1, learning_scheme='SGD')
    fc_trainer.train()
    # find correctly classified image.
    seed = 777
    image, label = get_correctly_classified_image(fc_trainer.model,
                                                  seed)
    plt.imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.title(f'Correctly classified as {label}')
    plt.savefig(os.path.join(RESULTS_DIR, f'correctly_classified_{label}.png'))
    # find adversarial examples
    adversarial_examples = []
    manipulated_indices = []
    adversarial_pixel_values = []
    adversarial_labels = []
    seed = 999
    while len(adversarial_examples) != 5:
        adversarial_example, manipulated_pixel_index, new_value, new_label = \
            log_and_get_adversarial_example(model, image, label, seed)
        seed += 1
        if manipulated_pixel_index in manipulated_indices:
            continue
        else:
            adversarial_examples.append(adversarial_example)
            manipulated_indices.append(manipulated_pixel_index)
            adversarial_pixel_values.append(new_value)
            adversarial_labels.append(new_label)

    # plot all adversarial examples in one figure
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.title(f'Original Image, label = {label}')
    for i in range(5):
        plt.subplot(2, 3, i + 1 + 1)
        plt.imshow(adversarial_examples[i].reshape(IMAGE_SIZE, IMAGE_SIZE))
        pixel_index = manipulated_indices[i]
        row, col = pixel_index // IMAGE_SIZE, pixel_index % IMAGE_SIZE
        new_label = adversarial_labels[i]
        l0 = cdist(adversarial_examples[i], image, metric='hamming').item()
        l2 = cdist(adversarial_examples[i], image, metric='euclidean').item()
        title = f"pixel index = ({row}, {col}), old, " \
                f"new label =({label, new_label}), \nL0={l0:.5f}, L2={l2:.2f}"
        plt.title(title)
        plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, 'all_adversarial_results.png'))

    # train another model
    another_model = AnotherModel()
    another_model_trainer = Trainer(another_model, learning_scheme='Adam', epochs=1)
    another_model_trainer.train()
    another_model = another_model_trainer.model

    # run the adversarial examples in the other model:
    another_model.eval()
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.title(f'Original Image, label = {label}')
    for i, sample in enumerate(adversarial_examples):
        plt.subplot(2, 3, i + 1 + 1)
        output = another_model(sample)
        _, prediction = torch.max(output.data, 1)
        plt.imshow(sample.reshape(IMAGE_SIZE, IMAGE_SIZE))
        plt.title(f'Second model, label = {prediction.item()}')
    plt.savefig(os.path.join(RESULTS_DIR,
                             'adversarial_examples_in_another_model.png'))


if __name__ == '__main__':
    main()

