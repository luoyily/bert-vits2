BertVits2 个人测试用副本（Fork于11.7日版本）（即修复日语Bert BUG后的版本）

原仓库：https://github.com/fishaudio/Bert-VITS2



## 训练

0. 配置环境，下载所需的Bert模型（见`bert/bert_models.json`），底模（见原仓库release）等。

1. 运行`config.py`生成配置，然后打开`config.yml`并填写`dataset_path`（可按需修改预处理中的文件路径，也可以按以下默认格式存放数据）

```
train.txt示例:
example.ogg|星奏|JP|「私が戻ってきたのはね。 もう一度、星の音を聞くためだよ」
folder1/example2.ogg|灰桜|JP|「みゅ？ わたしのお家……なのでしょうか？」
默认配置对应目录示例：
+-- Data
  +-- 你的数据集文件夹
    +-- audio
      |-- example.ogg
      +-- folder1
        |-- example2.ogg
    +-- filelist
      |-- eval.txt
      |-- train.txt
```

2. 运行`data_pack.py`预处理数据

3. 运行`train_debug.py`开始训练（暂时仅支持单卡）

   其中66行处 DataLoader的num workers等配置可按需自行调整

## 推理

待补充，推荐配合https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI使用

## 温馨提示

1. 预处理暂时只做了日语，其他可能之后补上
2. 预处理音频采样率相关以及add blank参数固定为了默认config.json中的参数，如有需要自行修改

## 关于

1. 此版本可能存在延后，不一定长期同步原版
2. 此版本主要为个人学习使用，正在施工中。目的为方便调试和阅读代码，训练等（可能）
3. 一般我们会本地测试后提交，但也可能存在BUG，如果遇到可发Issue。
