# Agricultural-Disease-Classification
背景

在keras的基础上写了一个 Ai Challenger 农作物竞赛的 baseline。

比赛地址（competition address）：[农作物病害检测](https://challenger.ai/competition/pdr2018)

完整代码地址（full code address）：[plants_disease_detection](https://github.com/bochuanwu/Agricultural-Disease-Classification/)
*成绩（score）**：线上（online） 0.88658
最终测试集（final） testB top 20
### 1. 依赖（dependence）

    python3.5 keras2.2.2
### 2. 模型选择（model chosen）

模型尝试了 resnet50，densenet ，invectionV3，vgg
### 3. 进行了图像增强（image augmentation）
旋转（rotate）
翻转（flip）
模糊（blur）
随机裁剪（random crop）

### 4. 数据分析（data analysis）
![train](https://github.com/bochuanwu/Agricultural-Disease-Classification/tree/master/dataset/test.png)
