#batcha training
import numpy as np
from torchvision.datasets import MNIST

def normalize_data(data):
    return np.array(data) / 255.0

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        one_hot[i][labels[i]] = 1 
    return one_hot

def split_batches(data, labels, batch_size):
    indices = np.random.permutation(len(data))
    data_shuffled = data[indices]
    labels_shuffled = labels[indices]

    split_data = []
    split_labels = []
    
    for i in range(0, len(data), batch_size):
        j = min(i + batch_size, len(data))
        split_data.append(data_shuffled[i:j])
        split_labels.append(labels_shuffled[i:j])
    return split_data, split_labels

def forward_propagation(data, w, bias):
    results = np.dot(data, w) + bias.T
    results -= np.max(results, axis=1, keepdims=True)
    res_exp = np.exp(results)
    res_sum = np.sum(res_exp, axis=1, keepdims=True)
    res_sum[res_sum == 0] = 1e-12
    prob = res_exp / res_sum
    return prob

def crossentropy(predictions,labels):
    return -np.sum(labels * np.log(predictions + 1e-12), axis=1, keepdims=True)

def train(data, labels, w, bias, lr):
    prob = forward_propagation(data,w,bias)
    error = labels - prob
    delta_w = lr * np.dot(data.T,error)
    temp_bias = lr * np.sum(error, axis=0, keepdims=True).T
    return delta_w, temp_bias

def test_acc(data, labels, w, bias):
    prob = forward_propagation(data,w,bias)
    predictions = np.argmax(prob, axis=1, keepdims=True)
    labels  = np.argmax(labels, axis=1, keepdims=True)
    correct_predictions = np.sum(predictions == labels)
    return correct_predictions*100 / len(labels)

def download_mnist(is_train: bool):
    dataset = MNIST(root='./Homework2/data',
                transform=lambda x: np.array(x).flatten(),
                download=True,
                train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels


train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = normalize_data(train_X)
test_X = normalize_data(test_X)

train_Y = one_hot_encode(train_Y, 10)
test_Y = one_hot_encode(test_Y, 10)

batch_size = 100
epoch_num = 200
np.random.seed(42)
w = np.random.uniform(-0.01, 0.01, (784, 10))
bias = np.random.uniform(-0.01, 0.01, (10, 1))
lr = 0.0001

print("initial acc " + str(test_acc(test_X, test_Y, w, bias)))

for i in range(epoch_num):
    batches_X, batches_Y = split_batches(train_X, train_Y, batch_size)
    for batch_X, batch_Y in zip(batches_X, batches_Y):
        delta_w, temp_bias = train(batch_X, batch_Y, w, bias, lr)
        w = w + delta_w
        bias = bias + temp_bias
    print("epoch " + str(i+1) + " acc " + str(test_acc(test_X, test_Y, w, bias)))


