# 根據前面範例 做分析 並且根據 所需重要參數 做 預測
# 建立好預測模型後 針對某一變數 做調整 看defect數 會上升還是下降 藉以參考
# 只用兩層NN 做回歸預測
import os
import numpy as np
import pandas as pd
import time
# 繪圖
import matplotlib.pyplot as plt
import seaborn as sns
# 深度學習 套件 pytorch 
import torch.nn as nn
import torch
import torch.utils.data
# sklearn 套件 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
# feature select
from xgboost import XGBRegressor,plot_importance
import lightgbm as lgb

# 將要讀取的資料 放同一folder內 可以直接用 os.getCwd() 
file_name = "sample.csv"
title_name = "712_726" + "_L1_" + "Machine_" + "dType"
file_path = os.getcwd() + "\\" + file_name
df = pd.read_csv(file_path) # df : dataframe
print(df.shape) # 資料大小


# drop 掉 非參數的項目 名稱要打對不然會報錯
# 主要可以透過此分析 param1 如果有變動 可能對 某一參數也有影響
pd.set_option('precision',2)
plt.figure(figsize=(10,8))
sns.heatmap(df.drop(['Target','LotID','GlassID'], axis=1).corr(), square=True)
plt.suptitle("Pearson Correlation Heatmap") # 參數間彼此的相關性

# 異常defect數量 與參數間的相關性  PS: 這邊可以看到 df 是全域變數 所以這邊drop後就會少這些項
df.drop(['oper'],axis=1, inplace=True) # 處理前幾項非影響參數 ex: gID ,oper..
# 這邊lib會自動排除非數值的變數 所以主要處理掉 前幾項
corr_with_target = df.corr()['Target'].sort_values(ascending = False)
plt.figure(figsize=(14,8))
plt.title(title_name)
corr_with_target.drop("Target").plot.bar()
plt.show()

# 這個看數據再使用
# num_feature = df.dtypes[df.dtypes != "object"].index # 過濾出只有數值的變量(去掉非數值項目)
# print(len(df[num_feature].columns)) 
# df = df[num_feature].fillna(0)              # 用0填補空值
x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.2, random_state=0) # 通常會0.8比0.2做切分 ,這邊不使用亂數選取
print("訓練資料筆數 : "+ str(x_train.shape) + "; 驗證資料筆數" + str(x_test.shape)) # 




# nn model
class FC_model(nn.Module):
    def __init__(self, in_plane):
        super(FC_model, self).__init__()
        self.in_plane = in_plane # 輸入的變數數量

        self.fc1 = nn.Linear(self.in_plane, 128) # 第1層 net
        self.fc2 = nn.Linear(128, 32)            # 第2層 net , in 的數量要與上一層一致
        self.out = nn.Linear(32, 1)

        self.ReLU = nn.ReLU() # activation function, 將值限制再一定範圍, 一般會將數據做標準化 將預測的數值限制在一個範圍 這樣計算loss時比較好收斂
        self.pReLu = nn.PReLU()
    def forward(self, x):
        x = self.pReLu(self.fc1(x))
        x = self.ReLU(self.fc2(x))  

        out = self.ReLU(self.out(x))
        return out

# 訓練數據 格式轉換 np To tensor  ； x = data , y=label
# 注意一點 訓練資料的shape[1500, 36] 對應 label [1500] 實際上必須為[1500,1] 會多給一個維度
x_train = torch.from_numpy(np.array(x_train)).type(torch.FloatTensor)
x_test = torch.from_numpy(np.array(x_test)).type(torch.FloatTensor) # 在訓練的通常是做驗證集(validation data)
y_train = torch.from_numpy(np.array(y_train)).type(torch.FloatTensor) # [1500] > [1500,1]
y_train = y_train.unsqueeze(1) # 使用 unsqueeze 多添加一個維度 
y_test = torch.from_numpy(np.array(y_test)).type(torch.FloatTensor)
y_test = y_test.unsqueeze(1)
print(" 訓練集 == data shape:" , x_train.size() , "; label shape",y_train.size())
dataset = torch.utils.data.TensorDataset(x_train,y_train)
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=16, shuffle=True) # batch_size 不開太大 8,16,32,64 因為沒GPU 所以8或16
EPOCH = 20 # 跑的次數

model = FC_model(in_plane = x_train.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_MSE = nn.MSELoss()

for epoch in range(EPOCH):
    for n, (Data, Label) in enumerate(dataloader):
        optimizer.zero_grad() 
        
        pred = model(Data)
        loss = loss_MSE(pred, Label)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
         # 每一次 epoch 計算一下 驗證預測值
        print('epoch[{}], loss:{:.4f}'.format(epoch+1, loss.item()))
