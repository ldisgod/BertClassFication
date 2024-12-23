# 基于pytorch_bert的中文多标签分类
本项目是基于pytorch+bert的中文多标签文本分类，使用的数据集是百度的：《2020语言与智能技术竞赛：事件抽取任务》中的数据，其官网为：<a href="https://aistudio.baidu.com/aistudio/competition/detail/32">2020语言与智能技术竞赛：事件抽取任务</a>，在该项目中，数据存放在data/raw_data文件夹下。<br>

## 使用依赖
```python
torch==1.6.0
transformers==4.5.1
```
## 相关说明
--logs：存放日志<br>
--checkpoints：存放保存的模型<br>
--data：存放数据<br>
--utils：存放辅助函数<br>
--bert_config.py：相关配置<br>
--dataset.py：制作数据为torch所需的格式<br>
--preprocess.py：数据预处理成bert所需要的格式<br>
--models.py：存放模型代码
--main.py：主运行程序，包含训练、验证、测试、预测以及相关评价指标的计算<br>
要预先下载好预训练的bert模型，放在和该项目同级下的model_hub文件夹下，即：<br>
model_hub/bert-base-chinese/
相关下载地址：<a href="https://huggingface.co/bert-base-chinese/tree/main=">bert-base-chinese</a><br>
需要的是vocab.txt、config.json、pytorch_model.bin

## 一般步骤
先从preprocess.py中看起，里面有处理数据为bert所需格式的相关代码，相关运行结果会保存在logs下面的preprocess.log中。然后看dataset.py代码，里面就是制作成torch所需格式的数据集。感兴趣的可以继续看看models.py中模型建立的过程。最终的运行主函数在main.py中。在main.py中运行的结果会保存在logs下的main.log中。

## 运行
```python
python main.py \
--bert_dir="../model_hub/bert-base-chinese/" \
--data_dir="./data/final_data/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=65 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=128 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=5 \
--eval_batch_size=32 \
```
在main.py中有相关代码控制训练、验证、测试、预测，根据需要注释掉其他的。

## 结果
### 训练和验证（部分结果）
```
2021-07-16 15:07:46,448 - INFO - main.py - train - 70 - 【train】 epoch：4 step:1799/1870 loss：0.019321
2021-07-16 15:07:51,781 - INFO - main.py - train - 75 - 【dev】 loss：0.722948 accuracy：0.8344 micro_f1：0.8982 macro_f1：0.7839
2021-07-16 15:07:51,782 - INFO - main.py - train - 77 - ------------>保存当前最好的模型
```
### 测试
由于没有测试集，因此验证和测试使用的是同一个数据集：
```python
========进行测试========
2021-07-16 15:10:11,912 - INFO - main.py - <module> - 187 - 【test】 loss：0.754452 accuracy：0.8144 micro_f1：0.8879 macro_f1：0.7593
2021-07-16 15:10:11,957 - INFO - main.py - <module> - 189 -               precision    recall  f1-score   support

 财经/交易-出售/收购       1.00      0.88      0.93        24
    财经/交易-跌停       1.00      0.79      0.88        14
    财经/交易-加息       0.00      0.00      0.00         3
    财经/交易-降价       1.00      0.78      0.88         9
    财经/交易-降息       0.00      0.00      0.00         4
    财经/交易-融资       1.00      1.00      1.00        14
    财经/交易-上市       1.00      0.14      0.25         7
    财经/交易-涨价       1.00      0.20      0.33         5
    财经/交易-涨停       0.96      0.96      0.96        27
     产品行为-发布       0.97      0.95      0.96       150
     产品行为-获奖       0.93      0.81      0.87        16
     产品行为-上映       0.97      0.89      0.93        35
     产品行为-下架       1.00      0.83      0.91        24
     产品行为-召回       1.00      1.00      1.00        36
       交往-道歉       0.94      0.89      0.92        19
       交往-点赞       1.00      0.64      0.78        11
       交往-感谢       1.00      0.12      0.22         8
       交往-会见       1.00      0.92      0.96        12
       交往-探班       1.00      0.60      0.75        10
     竞赛行为-夺冠       0.84      0.84      0.84        56
     竞赛行为-晋级       0.94      0.88      0.91        33
     竞赛行为-禁赛       0.93      0.88      0.90        16
     竞赛行为-胜负       0.98      0.95      0.97       213
     竞赛行为-退赛       1.00      0.50      0.67        18
     竞赛行为-退役       1.00      0.91      0.95        11
     人生-产子/女       1.00      0.53      0.70        15
       人生-出轨       0.00      0.00      0.00         4
       人生-订婚       1.00      1.00      1.00         9
       人生-分手       1.00      0.93      0.97        15
       人生-怀孕       1.00      0.50      0.67         8
       人生-婚礼       0.00      0.00      0.00         6
       人生-结婚       1.00      0.72      0.84        43
       人生-离婚       1.00      0.88      0.94        33
       人生-庆生       1.00      0.94      0.97        16
       人生-求婚       1.00      0.67      0.80         9
       人生-失联       1.00      0.36      0.53        14
       人生-死亡       0.96      0.66      0.78       106
     司法行为-罚款       1.00      0.86      0.93        29
     司法行为-拘捕       0.97      0.95      0.96        88
     司法行为-举报       1.00      1.00      1.00        12
     司法行为-开庭       0.93      1.00      0.97        14
     司法行为-立案       0.90      1.00      0.95         9
     司法行为-起诉       0.92      0.57      0.71        21
     司法行为-入狱       0.88      0.78      0.82        18
     司法行为-约谈       0.97      0.97      0.97        32
    灾害/意外-爆炸       1.00      0.11      0.20         9
    灾害/意外-车祸       0.88      0.86      0.87        35
    灾害/意外-地震       1.00      0.86      0.92        14
    灾害/意外-洪灾       1.00      0.57      0.73         7
    灾害/意外-起火       0.91      0.78      0.84        27
  灾害/意外-坍/垮塌       1.00      0.40      0.57        10
    灾害/意外-袭击       0.82      0.56      0.67        16
    灾害/意外-坠机       1.00      0.92      0.96        13
     组织关系-裁员       1.00      0.79      0.88        19
   组织关系-辞/离职       0.99      0.96      0.97        71
     组织关系-加盟       0.97      0.78      0.86        41
     组织关系-解雇       1.00      0.46      0.63        13
     组织关系-解散       1.00      0.70      0.82        10
     组织关系-解约       0.00      0.00      0.00         5
     组织关系-停职       1.00      0.82      0.90        11
     组织关系-退出       0.93      0.64      0.76        22
     组织行为-罢工       1.00      0.88      0.93         8
     组织行为-闭幕       0.88      0.78      0.82         9
     组织行为-开幕       0.93      0.88      0.90        32
     组织行为-游行       1.00      0.78      0.88         9

   micro avg       0.96      0.82      0.89      1657
   macro avg       0.90      0.70      0.76      1657
weighted avg       0.95      0.82      0.87      1657
 samples avg       0.88      0.85      0.86      1657
```
## 预测
预测的结果没有保存在log中，在main.py中最下面一段代码就是的，自行运行即可。


