import torch
from model.transformer import ChatTransformer
from src.utils.tokenizer import Tokenizer
from config import Config

class Chatbot:
    def __init__(self):
        self.config = Config()
        self.tokenizer = Tokenizer(vocab_size=self.config.VOCAB_SIZE)
        self.tokenizer.load_vocab(self.config.VOCAB_PATH)  # 加载词汇表
        self.model = ChatTransformer(
            self.config.EMBED_SIZE,
            self.config.FFN_HIDDEN_SIZE,
            self.config.NUM_HEADS,
            self.config.NUM_LAYERS,
            self.config.VOCAB_SIZE,
            self.config.MAX_SEQ_LEN,
            self.config.DROPOUT
        ).to(self.config.DEVICE)
        # 确保模型权重移动到同一设备
        self.model.load_state_dict(torch.load(self.config.MODEL_SAVE_PATH, map_location=self.config.DEVICE))
        self.model.eval()
        print(f"模型设备：{next(self.model.parameters()).device}")

    def generate_response(self, input_text, top_k=50, top_p=0.9, max_length=100):
        input_ids = self.tokenizer.text_to_ids(input_text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.config.DEVICE)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs
            if len(logits.shape) != 2:
                logits = logits.view(-1, logits.size(-1))

            filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            _, next_token_ids = torch.topk(probs, k=1, dim=-1)
            next_token_ids = next_token_ids.squeeze(1)

            # 生成完整的序列
            generated_ids = []
            current_ids = input_ids.squeeze(1)
            generated_ids.append(current_ids.tolist())
            for _ in range(max_length):
                current_ids = torch.tensor([generated_ids[-1]]).to(self.config.DEVICE)
                outputs = self.model(current_ids)
                logits = outputs.view(-1, outputs.size(-1))
                filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                _, next_token_ids = torch.topk(probs, k=1, dim=-1)
                next_token_ids = next_token_ids.squeeze(1)
                generated_ids.append(next_token_ids.tolist()[0])
                if next_token_ids == self.tokenizer.get_token_id("[SEP]"):
                    break

            generated_ids = [id for ids in generated_ids for id in ids]
            response = self.tokenizer.ids_to_text(generated_ids)
            return response

    def top_k_top_p_filtering(self, logits, top_k=100, top_p=0.9):
        if len(logits.shape) != 2:
            logits = logits.view(-1, logits.size(-1))
        batch_size, vocab_size = logits.shape
        if top_k > 0:
            # 获取前top_k个索引
            _, indices = torch.topk(logits, k=top_k, dim=-1)
            # 创建一个mask，保留top_k个元素，其余设置为-∞
            mask = torch.zeros_like(logits).scatter_(-1, indices, 1.).bool()
            logits = logits.masked_fill(~mask, float('-inf'))
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # 确定超过top_p的位置
            mask = (cumulative_probs >= top_p)
            # 获取需要设置为-∞的索引
            sorted_indices_to_remove = sorted_indices[mask]
            # 确保sorted_indices_to_remove与logits具有相同的维度
            if len(sorted_indices_to_remove.shape) < len(logits.shape):
                sorted_indices_to_remove = sorted_indices_to_remove.unsqueeze(0)
            # 将这些索引的位置设置为-∞
            logits = logits.scatter_(-1, sorted_indices_to_remove, float('-inf'))
        return logits


if __name__ == "__main__":
    chatbot = Chatbot()
    print("Welcome to the Chatbot!")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

