# Ref :  https://blog.csdn.net/weixin_42608414/article/details/88560116

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor,plot_importance
import lightgbm as LGB
import torch.nn as nn
import torch
import torch.utils.data

# if read .csv
file_path = os.getcwd() + '\\train.csv'
df = pd.read_csv(file_path)
print(df.shape) # total (1460, 81)
# file_path = os.getcwd() + '\\train.xlsx'
# df = pd.read_excel(file_path)
# print(df.shape) # total (1460, 81)

#打印數據 可以看到有NaN 以及其他非數字的數據
print(df.head)

# 檢查 變數之間的彼此的相關熱力圖 可以注意到 只有數值的地方才會被抓到 ******
# pd.set_option('precision',2)
# plt.figure(figsize=(10,8))
# sns.heatmap(df.drop(['SalePrice'], axis=1).corr(), square=True)
# plt.suptitle("Pearson Correlation Heatmap")
# plt.show()

# 主要 : 變數與目標的相關性分析，從結果可以看到 那些變數與SalePrice 有正相關影響 **********
corr_with_target = df.corr()['SalePrice'].sort_values(ascending = False)
# plt.figure(figsize=(14,6))
# corr_with_target.drop("SalePrice").plot.bar()
# plt.show()

# 數據處理
df.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1, inplace=True) # 處理過多空值項目 與 Id處理過多空值項目 與 Id
num_feature = df.dtypes[df.dtypes != "object"].index                        # 過濾出只有數值的變量(去掉非數值項目)
print(len(df[num_feature].columns))         #原先81筆 處理後 剩下37筆
df = df[num_feature].fillna(0)              # 用0填補空值
# print(df.isnull().values.any())             # 檢查是否有NaN值
# for i in range(len(df[num_feature].columns)):
#     print(df[num_feature[i]].astype(str).str.contains('NA').any()) # 檢查是否有NA字串 其中 .astype(str)轉型

# 利用 sklearn lib 做數據的切分 訓練集(合)與驗證集(合)
x, y = df.drop(['SalePrice'],axis=1) ,df['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.2, random_state=0) # 通常會0.8比0.2做切分
print("訓練資料筆數 : "+ str(x_train.shape) + "; 驗證資料筆數" + str(x_test.shape)) # 
print(torch.from_numpy(np.array(y_train)).unsqueeze(1).size())

# # ===== 使用xgboost 找出重要參數 ===== ****
# xgb_model_1 = XGBRegressor()
# xgb_model_1.fit(x_train, y_train, verbose=False)
# y_train_pred_1 = xgb_model_1.predict(x_train)
# y_pred_1 = xgb_model_1.predict(x_test)
# print('xgb_model_1 train r2_score', r2_score(y_train_pred_1, y_train))
# print('xgb_model_1 test r2_score', r2_score(y_test, y_pred_1))
# train_mse_1 = mean_squared_error(y_train_pred_1, y_train)
# train_rmse_1 = np.sqrt(train_mse_1)
# print('xgb_model_1 train RMSE :' ,train_rmse_1)
# test_mse_1 = mean_squared_error(y_pred_1,y_test)
# test_rmse_1 = np.sqrt(test_mse_1)
# print('xgb_model_1 test RMSE :' , test_rmse_1)

# dic = xgb_model_1.get_booster().get_fscore()
# feature_import_1 = pd.DataFrame.from_dict(dic, orient='index', columns=['fscore1'])
# print('xgb model 1 feature num :', len(feature_import_1))
# plot_import_f_1 = feature_import_1.sort_values(by='fscore1', ascending=False)[:30]
# plot_import_f_1.plot(kind='bar', figsize=(12,7))
# plt.show()

# # 使用Learning Rate 並調整參數
# xgb_model_2 = XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=0, 
# subsample = 0.75, colsample_bytree=1, max_depth=7, n_jobs=-1)
# xgb_model_2.fit(x_train, y_train, verbose=False)
# y_train_pred_2 = xgb_model_2.predict(x_train)
# y_pred_2 = xgb_model_2.predict(x_test)
# print('xgb_model_2 train r2_score', r2_score(y_train_pred_2, y_train))
# print('xgb_model_2 test r2_score', r2_score(y_test, y_pred_2))
# # 這邊不探討 預測 主要看 重要參數

# # ===== 使用LightGBM 找出重要參數 ===== *****
# lgb_params = {'boosting_type':'gbdt',
#               'objective':'regression',       # 此為regression task 如果為分類 則做修改 再上網找doc.
#               'num_leaves':2**8-1,            # 網路推薦 2^網路推薦 2^max_depth -1 
#               'max_depth':8,
#               'learning_rate':0.05,           # 深度學習 重要參數之一 學習率
#               'verbose':-1}                   # 消 warming
# lgb_train_dataset = LGB.Dataset(x_train, y_train)
# lgb_model = LGB.train(lgb_params, lgb_train_dataset)
# LGB.plot_importance(lgb_model, max_num_features=30,figsize=(12,7),title ="lightGBM import Feature" )
# plt.show()

# 透過xgboost 以及 lightGBN 可知 共通最重要參數為 'LotArea'



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