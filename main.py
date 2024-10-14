from torchvision.datasets import MNIST
import numpy as np
import torch

def one_hot_encoding(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

def process_data(dataset):
    images = np.array(dataset.data)
    labels = np.array(dataset.targets)

    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    images_tensor /= 255.0
    images_tensor = images_tensor.view(-1, 28 * 28)
    one_hot_labels = one_hot_encoding(labels_tensor)

    return images_tensor, one_hot_labels

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weight1 = torch.randn(input_size, hidden_size) * 0.1
        self.bias1 = torch.zeros(hidden_size)
        self.weight2 = torch.randn(hidden_size, output_size) * 0.1
        self.bias2 = torch.zeros(output_size)
    def relu(self, x):
        return torch.maximum(x, torch.tensor(0.0))
    def relu_derivate(self, x):
        return (x > 0).float()
    def softmax(self, x):
        exp_z = torch.exp(x)
        return exp_z / torch.sum(exp_z, dim=1, keepdim=True)
    def forward(self, input):
        self.z1 = input @ self.weight1 + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.weight2 + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2
    def backward(self, x, y, learning_rate = 0.01):
        nr_samples = y.shape[0]

        output_loss_gradient = self.a2 - y
        weight2_gradient = self.a1.t() @ output_loss_gradient / nr_samples
        bias2_gradient = output_loss_gradient.mean(dim=0)

        self.weight2 -= learning_rate * weight2_gradient
        self.bias2 -= learning_rate * bias2_gradient

        hidden_loss_gradient = output_loss_gradient @ self.weight2.t() * self.relu_derivate(self.z1)
        weight1_gradient = x.t() @ hidden_loss_gradient / nr_samples
        bias1_gradient = hidden_loss_gradient.mean(dim=0)

        self.weight1 -= learning_rate * weight1_gradient
        self.bias1 -= learning_rate * bias1_gradient

def train(model, images, one_hot_labels, epochs=100, lr=0.1, batch_size=64):
    num_samples = images.shape[0]
    indices = np.arange(num_samples)

    for epoch in range(epochs):
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_images = images[batch_indices]
            batch_labels = one_hot_labels[batch_indices]

            y_pred = model.forward(batch_images)

            loss = torch.nn.functional.cross_entropy(y_pred, batch_labels)
            model.backward(batch_images, batch_labels, lr)

        predictions = torch.argmax(y_pred, dim=1)
        accuracy = (predictions == torch.argmax(batch_labels, dim=1)).float().mean()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}')


train_dataset = MNIST(root="./data", train=True, download=True, transform=None)
test_dataset = MNIST(root="./data", train=False, download=True, transform=None)

train_images, train_labels = process_data(train_dataset)
test_images, test_labels = process_data(test_dataset)

input_size = 28 * 28
hidden_size = 128
output_size = 10
model = MLP(input_size, hidden_size, output_size)

train(model, train_images, train_labels, epochs=100, lr=0.01, batch_size=64)

def evaluate(model, test_images, test_labels):
    with torch.no_grad():
        y_pred = model.forward(test_images)
        loss = torch.nn.functional.cross_entropy(y_pred, test_labels)
        predictions = torch.argmax(y_pred, dim=1)
        accuracy = (predictions == torch.argmax(test_labels, dim=1)).float().mean()
        print(f'Test Loss: {loss.item()}, Test Accuracy: {accuracy.item()}')

evaluate(model, test_images, test_labels)