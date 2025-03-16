import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import pdfplumber
from peft import PeftModel, PeftConfig
import json

def load_document(file_path):
    if file_path.endswith('.pdf'):
        document = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                document += page.extract_text()
                
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            document = f.read()
    else:
        raise ValueError("Unsupported file format. Only txt and pdf are supported.")
    return document



def read_data(query_data_path):
    with open(query_data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    return questions



# 加载指令微调后的大模型
# model_name_or_path = "/hdd/junxuanl/nlp2024/Qwen2.5-0.5B-SFT_cos_2e-5_wct"


# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()
model_name_or_path = "/hdd/junxuanl/nlp2024/Qwen2.5-3B"
lora_adapter_dir = "/hdd/junxuanl/nlp2024/Qwen-2.5-3B-Alpaca-LoRA"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
model.merge_and_unload()
model.eval()



# 选择 GPU 设备编号
gpu_device_number = 7
device = torch.device(f"cuda:{gpu_device_number}" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置模型支持的最大长度
max_length = model.config.max_position_embeddings


# 加载文本嵌入模型
embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)


questions= read_data(query_data_path="/hdd/junxuanl/nlp2024/questions.json")


# 从文件读取文档内容
document = load_document('knowledge.pdf')

# 对文档进行预处理，分割成句子
sentences = re.split(r'[.。！!?？]', document)
sentences = [s.strip() for s in sentences if s.strip()]

# 嵌入文档句子
embeddings = embedder.encode(sentences, convert_to_tensor=True,normalize_embeddings=True).cpu().numpy()

# 构建向量数据库
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def chat():
    # 初始化对话历史
    conversation_history = []
    conversation_history_without_document = []
    print("欢迎使用聊天机器人！输入 \\quit 结束会话，输入 \\newsession 开启新的对话。")
    while True:
        user_input = input("user: ")

        if user_input == "\\quit":
            print("会话结束，再见！")
            break
        elif user_input == "\\newsession":
            conversation_history.clear()
            conversation_history_without_document.clear()
            print("新的对话已开启。")
        else:
            # 嵌入用户问题
            question_embedding = embedder.encode([user_input], convert_to_tensor=True).cpu().numpy()

            # 检索相关句子
            distances, indices = index.search(question_embedding, k=5)
            retrieved_sentences = [sentences[i] for i in indices[0]]

            # 拼接历史输入
            history_text = ""
            if conversation_history:
                if len(conversation_history) > max_length:
                    conversation_history = conversation_history[-max_length:]
                for turn in conversation_history:
                    history_text += f"user: {turn['user']}\nbot: {turn['bot']}\n"
            input_text = history_text + " ".join(retrieved_sentences) + f" user: {user_input}\n"

            history_text = ""    
            if conversation_history_without_document:
                if len(conversation_history_without_document) > max_length:
                    conversation_history_without_document = conversation_history_without_document[-max_length:]
                for turn in conversation_history_without_document:
                    history_text += f"user: {turn['user']}\nbot: {turn['bot']}\n"
            input_text_without_document = history_text + f" user: {user_input}\n"
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ]
            messages_without_document = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text_without_document},
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            text_without_document = tokenizer.apply_chat_template(
                messages_without_document,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer([text], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # 将所有输入张量移动到模型所在设备

            inputs_without_document = tokenizer([text_without_document], return_tensors="pt")
            inputs_without_document = {k: v.to(device) for k, v in inputs_without_document.items()}  # 将所有输入张量移动到模型所在设备

            try:
                generated_ids = model.generate(inputs['input_ids'], max_new_tokens=256, num_beams=4,
                                               attention_mask=inputs['attention_mask'],
                                               pad_token_id=tokenizer.eos_token_id)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'],
                                                                                          generated_ids)]
                reply = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                generated_ids_without_document = model.generate(inputs_without_document['input_ids'], max_new_tokens=256, num_beams=4,
                                               attention_mask=inputs_without_document['attention_mask'],
                                               pad_token_id=tokenizer.eos_token_id)
                generated_ids_without_document = [output_ids_without_document[len(input_ids_without_document):] for input_ids_without_document, output_ids_without_document in zip(inputs_without_document['input_ids'],
                                                                                          generated_ids_without_document)]
                reply_without_document = tokenizer.batch_decode(generated_ids_without_document, skip_special_tokens=True)[0]

            except Exception as e:
                print(f"机器人生成回复时出现异常: {e}")
                reply = "很抱歉，我暂时无法生成回复，请稍后再试。"

            # 打印回复
            print(f"bot_with_document: {reply}")
            print(f"bot_without_document: {reply_without_document}")

            # 更新对话历史
            conversation_history.append({"user": user_input, "bot": reply})
            conversation_history_without_document.append({"user": user_input, "bot": reply_without_document})


def evaluate():
    # 初始化对话历史
    conversation_history = []
    conversation_history_without_document = []
    with open("evaluation_output.txt", "w", encoding="utf-8") as f:
        for question in questions:
            user_input = question['question']
            f.write(f"Question: {user_input}\n\n")
            # 嵌入用户问题
            question_embedding = embedder.encode([user_input], convert_to_tensor=True).cpu().numpy()

            # 检索相关句子
            distances, indices = index.search(question_embedding, k=6)
            retrieved_sentences = [sentences[i] for i in indices[0]]
            f.write(f"Context: {retrieved_sentences}\n\n")
            # 拼接历史输入
            history_text = ""
            if conversation_history:
                if len(conversation_history) > max_length:
                    conversation_history = conversation_history[-max_length:]
                for turn in conversation_history:
                    history_text += f"user: {turn['user']}\nbot: {turn['bot']}\n"
            input_text = history_text + " ".join(retrieved_sentences) + f" user: {user_input}\n"

            history_text = ""    
            if conversation_history_without_document:
                if len(conversation_history_without_document) > max_length:
                    conversation_history_without_document = conversation_history_without_document[-max_length:]
                for turn in conversation_history_without_document:
                    history_text += f"user: {turn['user']}\nbot: {turn['bot']}\n"
            input_text_without_document = history_text + f" user: {user_input}\n"

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ]
            messages_without_document = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text_without_document},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            text_without_document = tokenizer.apply_chat_template(
                messages_without_document,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer([text], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # 将所有输入张量移动到模型所在设备

            inputs_without_document = tokenizer([text_without_document], return_tensors="pt")
            inputs_without_document = {k: v.to(device) for k, v in inputs_without_document.items()}  # 将所有输入张量移动到模型所在设备

            try:
                generated_ids = model.generate(inputs['input_ids'], max_new_tokens=256, num_beams=5,
                                               attention_mask=inputs['attention_mask'],
                                               pad_token_id=tokenizer.eos_token_id)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'],
                                                                                          generated_ids)]
                reply = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                generated_ids_without_document = model.generate(inputs_without_document['input_ids'], max_new_tokens=256, num_beams=5,
                                               attention_mask=inputs_without_document['attention_mask'],
                                               pad_token_id=tokenizer.eos_token_id)
                generated_ids_without_document = [output_ids_without_document[len(input_ids_without_document):] for input_ids_without_document, output_ids_without_document in zip(inputs_without_document['input_ids'],
                                                                                          generated_ids_without_document)]
                reply_without_document = tokenizer.batch_decode(generated_ids_without_document, skip_special_tokens=True)[0]

            except Exception as e:
                print(f"机器人生成回复时出现异常: {e}")
                reply = "很抱歉，我暂时无法生成回复，请稍后再试。"
                reply_without_document = "很抱歉，我暂时无法生成回复，请稍后再试。"

            # 打印回复
            f.write(f"bot_with_document: {reply}\n\n")
            f.write(f"bot_without_document: {reply_without_document}\n")
            f.write("-----------------------------------\n")
            # 更新对话历史
            conversation_history.append({"user": user_input, "bot": reply})
            conversation_history_without_document.append({"user": user_input, "bot": reply_without_document})

if __name__ == "__main__":
    # chat()
    evaluate()
    
    
    
    
