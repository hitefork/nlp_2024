# 自然语言处理arxiv小组大作业说明

本次大作业实现了三个bonus功能，即LoRA微调、RAG实现和角色扮演聊天机器人的实现。
因为原有模板对于中文的支持能力有限，我们改用了XeLaTeX进行排版，
paper.pdf是我们的正文，而papaer_use_template.pdf使用模板编译，用于证明原有内容没有超过页数限制。

## 文件结构
```bash
./nlp_2024/
├── README.md
├── codes
│   ├── bot
│   │   ├── chatbot
│   │   │   └── main.py
│   │   └── roleplay
│   │       ├── chat.py
│   │       └── prompt_factory.py
│   ├── lora
│   │   ├── finetune.py
│   │   ├── finetune_origin.yaml
│   │   └── merge_lora.yaml
│   └── rag
│       ├── knowledge.pdf
│       ├── questions.json
│       └── rag.py
├── paper.pdf
└── paper_use_template.pdf

6 directories, 12 files
```



## 运行方法

### LORA

首先是利用llamafactory-cli进行微调和合并的版本
```bash
cd ./codes/lora
llamafactory-cli train ./finetune_origin.yaml
...
llamafactory-cli merge ./merge_lora.yaml
```

然后是手动实现的版本
```bash
python3 ./finetune.py
```
[LoRA adapter 交大云盘路经](https://jbox.sjtu.edu.cn/l/x1kiKQ)


### RAG


```bash
cd ./codes/rag
python3 ./bonus_2.py
```
文档连接

### 聊天机器人

```bash
cd ./codes/bot
# version1_chatbot
cd ./chatbot
python3 main.py
# version2_roleplay_chatbot
cd ../roleplay
python3 chat.py
```

