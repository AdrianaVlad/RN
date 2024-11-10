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

def forward_propagation(data, w_ih, w_ho, bias_h, bias_o, training):
    hidden_input = np.dot(data, w_ih) + bias_h.T
    hidden_output = np.maximum(0, hidden_input)

    if training:
        mask = np.random.binomial(1, 1 - 0.25, size=hidden_output.shape)
        hidden_output *= mask
        hidden_output /= (1 - 0.25)


    results = np.dot(hidden_output, w_ho) + bias_o.T
    #results -= np.max(results, axis=1, keepdims=True)
    res_exp = np.exp(results)
    res_sum = np.sum(res_exp, axis=1, keepdims=True)
    #res_sum[res_sum == 0] = 1e-12
    prob = res_exp / res_sum
    
    return prob, hidden_output

def crossentropy(predictions,labels):
    return -np.sum(labels * np.log(predictions + 1e-12), axis=1, keepdims=True)

def train(data, labels, w_ih, w_ho, bias_h, bias_o, lr):
    prob, hidden_output = forward_propagation(data, w_ih, w_ho, bias_h, bias_o, True)

    error_o = labels - prob
    delta_w_ho = lr * np.dot(hidden_output.T,error_o) / 100
    temp_bias_o = lr * np.sum(error_o, axis=0, keepdims=True).T / 100

    error_h = hidden_output > 0
    error_h = error_h * np.dot(error_o, w_ho.T)
    delta_w_ih = lr * np.dot(data.T, error_h) / 100
    temp_bias_h = lr * np.sum(error_h, axis=0, keepdims=True).T / 100

    return delta_w_ih, delta_w_ho, temp_bias_h, temp_bias_o 

def test_acc(data, labels, w_ih, w_ho, bias_h, bias_o):
    prob, hidden_output = forward_propagation(data, w_ih, w_ho, bias_h, bias_o, False)
    predictions = np.argmax(prob, axis=1, keepdims=True)
    labels  = np.argmax(labels, axis=1, keepdims=True)
    correct_predictions = np.sum(predictions == labels)
    return correct_predictions*100 / len(labels)

def download_mnist(is_train: bool):
    dataset = MNIST(root='../Homework2/data',
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
epoch_num = 300
np.random.seed(42)
w_ih = np.random.uniform(-0.01, 0.01, (784, 100))
bias_h = np.random.uniform(-0.01, 0.01, (100, 1))
w_ho = np.random.uniform(-0.01, 0.01, (100, 10))
bias_o = np.random.uniform(-0.01, 0.01, (10, 1))
lr = 0.01
best = 0
wait = 10

print("initial acc " + str(test_acc(test_X, test_Y, w_ih, w_ho, bias_h, bias_o)))

for i in range(epoch_num):
    batches_X, batches_Y = split_batches(train_X, train_Y, batch_size)
    for batch_X, batch_Y in zip(batches_X, batches_Y):
        delta_w_ih, delta_w_ho, temp_bias_h, temp_bias_o = train(batch_X, batch_Y, w_ih, w_ho, bias_h, bias_o, lr)
        w_ih = w_ih + delta_w_ih
        w_ho = w_ho + delta_w_ho
        bias_h = bias_h + temp_bias_h
        bias_o = bias_o + temp_bias_o
    acc = test_acc(test_X, test_Y, w_ih, w_ho, bias_h, bias_o)
    acc_rounded = round(acc, 2)
    print("epoch " + str(i+1) + " acc " + str(acc) +" wait " + str(wait))
    if(best >= acc_rounded):
        wait -= 1
    else:
        best = acc
        wait = 10
    if(wait == 0):
        wait = 10
        lr /= 2
        print("adjusted lr: " + str(lr))
