# NLP-modeltest_bert_based

NLP作业 在三个下游任务上实现了测试与微调模型
使用时路径为Bert_forxxx文件夹下运行
# 1.模型下载
使用的模型在Bert_forxxx文件夹下有标出，已经添加vocab与config文件夹，pytorch.bin可对应模型名称从hugging face官网搜索下载
modelset文件为临时存的model参数文件，调试方便
# 2.运行
NER任务运行Bert_forNER/bert_ner/bert_ner.py
QA任务运行Bert_forQA/train.py实现训练，predict.py实现问答
Seqclassification任务运行Bert_forSeq/classifier.py
# 3.模型切换
修改参数MODEL_TYPE，对应修改读取模型的路径，在pretrained后
# 4.可能有问题
Seqclassification任务搭载Roberta模型时有时会报sizemismatch错误，可以使用test.py文件解决，但我的显卡内存不足，因此我调整了一下RobertaConfig下的相关参数，成功运行
