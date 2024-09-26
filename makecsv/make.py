import pandas as pd
from sklearn.datasets import fetch_openml

# 下載 MNIST 數據集
mnist = fetch_openml('mnist_784', version=1)
data, target = mnist['data'], mnist['target']

# 將數據和標籤合併為一個 DataFrame
df = pd.DataFrame(data)
df['label'] = target

# 保存為 CSV 檔案
csv_file = 'mnist.csv'
df.to_csv(csv_file, index=False)

print(f'MNIST 數據集已保存為 {csv_file}')