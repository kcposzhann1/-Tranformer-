# Transformer-Based Customer Service Chatbot

这是一个基于 Transformer 架构的智能客服聊天机器人项目。该项目旨在训练一个模型，使其能够根据客户问题生成合适的客服回答。

## 项目结构

- `train.py`: 模型训练脚本
- `chat.py`: 聊天机器人推理脚本
- `test.py`: 测试分词功能的脚本
- `config.py`: 项目配置文件
- `data.py`: 数据加载和预处理模块
- `tokenizer.py`: 分词器模块
- `chatbot.py`: 聊天机器人类
- `transformer.py`: Transformer 模型定义
- `train.xlsx`: 训练数据文件

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/your-username/transformer-chatbot.git
   cd transformer-chatbot
## Tips

1.本项目中构建了一个简化的transformer模型，实现了相应的MultiHeadAttention、positional_encoding
以及前反馈层，搭配相应数据集训练可以初步达到要求，但在对话的生成质量以及多样化方面均有所欠缺。

2.如有需要可以将数据集替换为更大的对话数据，并调整模型参数以提高模型对话能力
