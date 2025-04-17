import jieba

text = "请问这款手机电池续航能力如何，最近发现电量消耗得特别快"
words = jieba.lcut(text)
print("分词结果:", words)
