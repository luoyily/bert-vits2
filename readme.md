BertVits2 个人测试用副本（Fork于11.7日版本）（即修复日语Bert BUG后的版本）

原仓库：https://github.com/fishaudio/Bert-VITS2



## 训练

1. 配置并运行`data_pack.py`预处理数据（数据格式与要求见其中注释）
2. 配置`train_debug.py`运行即可（此文件仅支持单卡）

## 推理

待补充，推荐配合https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI使用

## 温馨提示

1. 此版本可能存在延后，不一定长期同步原版
2. 预处理暂时只做了日语，其他可能之后补上
3. 此版本主要为个人学习使用，正在施工中。目的为方便调试和阅读代码，训练等（可能）
4. 一般我们会本地测试后提交，但也可能存在BUG，如果遇到可发Issue。
