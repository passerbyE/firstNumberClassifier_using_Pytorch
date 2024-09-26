import torch
import torch.nn as nn
import pandas as pd

raw_df = pd.read_csv('mnist.csv')


#拿標籤
label = raw_df['label'].values
raw_df = raw_df.drop(['label'], axis=1)
#拿特徵
feature = raw_df.values

#訓練跟測試資料 (80%訓練, 20%測試)
train_feature = feature[:int(len(feature)*0.8)]
train_label = label[:int(len(label)*0.8)]

test_feature = feature[int(len(feature)*0.8):]
test_label = label[int(len(label)*0.8):]

train_feature = torch.tensor(train_feature).to(torch.float).cuda()
train_label = torch.tensor(train_label).to(torch.long).cuda()

test_feature = torch.tensor(test_feature).to(torch.float).cuda()
test_label = torch.tensor(test_label).to(torch.long).cuda()


#模型本體
data = torch.rand(1, 784)
model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()
)
model = model.cuda()

#損失函數(交叉熵)
lossfunction = nn.CrossEntropyLoss()
#梯度下降 可以講優化器，是怎麼下山的方法
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

#訓練輪數
for i in range(500):
    #清空有話器的梯度
    optimizer.zero_grad()
    #訓練
    predict = model(train_feature)
    result = torch.argmax(predict, axis=1)
    train_cc = torch.mean((result == train_label).to(torch.float))
    #算損失
    loss = lossfunction(predict, train_label)
    #反向傳播
    loss.backward()
    #梯度下降
    optimizer.step()

    print(f'第{i}輪的損失是= {loss.item()}\t\t正確率是= {train_cc.item()*100}%')
    
    optimizer.zero_grad()
    predict = model(test_feature)
    result = torch.argmax(predict, axis=1)
    test_cc = torch.mean((result == test_label).to(torch.float))
    loss = lossfunction(predict, test_label)
    print(f'測試第{i}輪的損失是= {loss.item()}\t\t正確率是= {test_cc.item()*100}%')

torch.save(model.state_dict(), './modelForOneT.pt')