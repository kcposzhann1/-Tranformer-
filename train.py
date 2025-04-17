import pandas as pd
from src.utils.data import ChatDataset
from src.model.transformer import ChatTransformer
from src.config import Config
from src.utils.tokenizer import Tokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    losses = []
    for batch in tqdm(dataloader, desc="Training"):
        client = batch["client"].to(device)
        server = batch["server"].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(client)

        # Reshape outputs and targets for loss calculation
        outputs_reshape = outputs.view(-1, outputs.size(-1))
        targets_reshape = server.view(-1)

        # Calculate loss (忽略Padding损失)
        loss = criterion(outputs_reshape, targets_reshape)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        losses.append(loss.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, losses


def main():
    config = Config()

    # 加载数据
    try:
        data = pd.read_excel(config.DATA_PATH)

        # 读取未分词的客户和客服对话内容
        client_contents = data["【中文】客户对话内容"].tolist()
        server_contents = data["【中文】客服对话内容"].tolist()
        # 这里假设我们只使用未分词的内容进行分词和生成词汇表
        texts = client_contents + server_contents

        print("Data loaded successfully.")
        print("Sample data from client:", client_contents[:5])
        print("Sample data from server:", server_contents[:5])
        print("Combined sample data:", texts[:5])
        if len(texts) == 0:
            print("Error: No training data found!")
            return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # 初始化 tokenizer
    tokenizer = Tokenizer(config.VOCAB_SIZE)
    # 使用未分词的内容生成词汇表
    tokenizer.build_vocab(texts=texts, is_tokenized=False)
    print("Vocabulary built successfully.")
    print("Vocabulary size:", len(tokenizer.vocab))

    # 保存词汇表
    tokenizer.save_vocab(config.VOCAB_PATH)
    print(f"Vocabulary saved to {config.VOCAB_PATH}")

    # 获取PAD的id
    pad_index = tokenizer.get_token_id("[PAD]")
    print("PAD index:", pad_index)

    # 获取UNK的id
    unk_index = tokenizer.get_token_id("[UNK]")
    print("UNK index:", unk_index)

    # 创建数据集和数据加载器
    # 这里需要修改ChatDataset类，以处理未分词的数据
    # 假设现在使用未分词的数据进行分词处理
    dataset = ChatDataset(config.DATA_PATH, tokenizer, config.MAX_SEQ_LEN, use_unsegmented=True)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, collate_fn=dataset.collate_fn)

    # 模型、优化器和损失函数
    model = ChatTransformer(
        config.EMBED_SIZE,
        config.FFN_HIDDEN_SIZE,
        config.NUM_HEADS,
        config.NUM_LAYERS,
        config.VOCAB_SIZE,
        config.MAX_SEQ_LEN,
        config.DROPOUT
    ).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = CrossEntropyLoss(ignore_index=pad_index)

    # 学习率调度器
    total_steps = len(dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


    # 训练
    all_losses = []
    for epoch in range(config.EPOCHS):
        avg_loss, losses = train(model, dataloader, optimizer, scheduler, criterion, config.DEVICE)
        all_losses.extend(losses)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses)
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # 保存词汇表
    tokenizer.save_vocab(config.VOCAB_PATH)
    print(f"Vocabulary saved to {config.VOCAB_PATH}")


if __name__ == "__main__":
    main()
