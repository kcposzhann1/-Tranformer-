import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len, use_unsegmented=False):
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_unsegmented = use_unsegmented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_unsegmented:
            # 使用未分词的客户和客服对话内容
            client_text = self.data.iloc[idx]["【中文】客户对话内容"]
            server_text = self.data.iloc[idx]["【中文】客服对话内容"]
        else:
            # 使用分词后的客户和客服对话
            client_text = self.data.iloc[idx]["【中文】客户对话分词"]
            server_text = self.data.iloc[idx]["【中文】客服对话分词"]

        # 使用 tokenizer 进行分词并转换为id
        client_ids = self.tokenizer.text_to_ids(client_text)
        server_ids = self.tokenizer.text_to_ids(server_text)

        # 填充或截断
        if len(client_ids) > self.max_seq_len:
            client_ids = client_ids[:self.max_seq_len]
        else:
            pad_length = self.max_seq_len - len(client_ids)
            pad = [self.tokenizer.get_token_id("[PAD]")] * pad_length
            client_ids += pad

        if len(server_ids) > self.max_seq_len:
            server_ids = server_ids[:self.max_seq_len]
        else:
            pad_length = self.max_seq_len - len(server_ids)
            pad = [self.tokenizer.get_token_id("[PAD]")] * pad_length
            server_ids += pad

        return {
            "client": torch.tensor(client_ids),
            "server": torch.tensor(server_ids)
        }

    def collate_fn(self, batch):
        clients = [b["client"] for b in batch]
        servers = [b["server"] for b in batch]

        clients_padded = torch.nn.utils.rnn.pad_sequence(
            clients,
            batch_first=True,
            padding_value=self.tokenizer.get_token_id("[PAD]")
        )
        servers_padded = torch.nn.utils.rnn.pad_sequence(
            servers,
            batch_first=True,
            padding_value=self.tokenizer.get_token_id("[PAD]")
        )

        return {
            "client": clients_padded,
            "server": servers_padded
        }







