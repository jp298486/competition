# competition2021_04-07_release(繁體中文影像辨識)
## 代碼更新中
Result

![image](https://github.com/jp298486/deep_learning_with_python/blob/main/Competition/competition2021_04-07_release/image/final_result.jpg)

```
根據baseline_with_Resnet_colab.ipynb結果
針對分類問題模型最後一層使用LogSoftmax
並且對於dataset的處理使用補0(padding)與插值的方式為實驗
影像大小縮放基本為64x64
對於32*n n>2以上的shape則加入attention機制
```