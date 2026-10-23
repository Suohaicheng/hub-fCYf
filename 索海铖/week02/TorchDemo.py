import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MltClsModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MltClsModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = torch.softmax
        # self.loss = nn.functional.mse_loss 
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):    
        x = self.linear(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
        
def build_sample():
    x = np.random.random(5)
    y = np.zeros_like(x)
    y[x.argmax()] = 1

    return x, y

def build_dataset(total_sample_num):
    X = []; Y = []

    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


def predict(model_path, input_vec):
    input_size = 5
    model = MltClsModel(input_size, input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        res = model.forward(torch.FloatTensor(input_vec))

    # print(res)
    for i in res:
        print(np.sum(i.numpy()))


def main():
    epoch_num = 20 ; batch_sz = 20; train_samples = 5000; input_size = 5; output_size = 5
    learning_rate = 0.01 

    model = MltClsModel(input_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train_x, train_y = build_dataset(train_samples)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_ix in range(train_samples // batch_sz):
            x = train_x[batch_ix * batch_sz : (batch_ix + 1) * batch_sz]
            y = train_y[batch_ix * batch_sz : (batch_ix + 1) * batch_sz]

            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n 第 %d 轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))                 

    torch.save(model.state_dict(), "model.bin")

def test_build_dataset():
    x, y = build_dataset(20)
    for i in range(20):
        print(f"{x[i]} --> {y[i]}")


def test_predict():    
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                        [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                        [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                        [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
                 
    predict("model.bin", test_vec)

if __name__ == '__main__':
    # test_build_dataset()
    # main()
    test_predict()
    