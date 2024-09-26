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

params = torch.load('./modelForOneT.pt')
model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()
).cuda()
model.load_state_dict(params)

star = 100
long = 10000

test_data = test_feature[star:long]
test_label = test_label[star:long]

predict = model(test_data)
result = torch.argmax(predict, axis=1)

if long - star < 10:
    print(result)
    print(test_label)


whoitWork = torch.mean((result == test_label).to(torch.float))
print('正確率= ', round(whoitWork.item()*100), '%')