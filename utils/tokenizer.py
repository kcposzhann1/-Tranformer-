# 修改后的tokenizer.py

import jieba
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.id_to_token = {}
        self.token_to_id = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3

    def build_vocab(self, texts=None, is_tokenized=False):
        if texts is None:
            texts = []
        all_words = []
        for text in texts:
            if isinstance(text, str):
                if is_tokenized:
                    words = text.split("|")
                    words = [w.strip() for w in words if w.strip() != '']
                    all_words.extend(words)
                else:
                    words = jieba.lcut(text)
                    all_words.extend(words)
        if not all_words:
            raise ValueError("所有单词都为空，无法生成词汇表")
        word_freq = Counter(all_words)
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        for word, _ in sorted_words:
            if len(vocab) >= self.vocab_size:
                break
            if word not in vocab:
                vocab[word] = len(vocab)
        self.vocab = vocab
        self.token_to_id = {token: idx for token, idx in self.vocab.items()}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        return self.vocab

    def text_to_ids(self, text):
        # 使用 jieba 进行分词
        tokens = jieba.lcut(text)
        ids = [self.get_token_id(token) for token in tokens]
        return ids

    def ids_to_text(self, ids):
        return " ".join([self.id_to_token.get(i, "[UNK]") for i in ids])

    def save_vocab(self, file_path):
        print("开始保存词汇表到文件...")
        print("词汇表中的词汇数量为:", len(self.vocab))
        for token, idx in self.vocab.items():
            print(f"{token}: {idx}")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for token, id in self.vocab.items():
                    f.write(f"{token}\t{id}\n")
            print(f"词汇表已成功保存到 {file_path}")
        except Exception as e:
            print(f"保存词汇表到 {file_path} 时发生错误：{str(e)}")

    def load_vocab(self, file_path):
        vocab = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    print(f"警告：第 {line_num} 行是空行，跳过...")
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"错误：第 {line_num} 行格式不正确，内容为：{line}")
                    print("期望格式为 'token\tid'")
                    raise ValueError(f"词汇表文件格式错误，第 {line_num} 行")
                token, id = parts
                if not token or not id:
                    print(f"错误：第 {line_num} 行包含空字段，内容为：{line}")
                    raise ValueError(f"词汇表文件包含空字段，第 {line_num} 行")
                try:
                    id = int(id)
                except ValueError:
                    print(f"错误：第 {line_num} 行的id不是整数，内容为：{line}")
                    raise ValueError(f"词汇表文件中id必须是整数，第 {line_num} 行")
                if token in vocab:
                    print(f"警告：第 {line_num} 行的token '{token}' 已经存在于词汇表中")
                vocab[token] = id
        self.vocab = vocab
        self.id_to_token = {id: token for token, id in self.vocab.items()}
        self.token_to_id = {token: id for token, id in self.vocab.items()}

    def get_token_id(self, token):
        return self.token_to_id.get(token, self.token_to_id.get("[UNK]", 1))
