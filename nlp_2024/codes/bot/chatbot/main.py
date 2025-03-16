from threading import Thread
from typing import *
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria
from peft import PeftModel

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_chars, tokenizer):
        self.tokenizer = tokenizer
        self.stop_chars = stop_chars

    def __call__(self, input_ids, scores, **kwargs):
        token_id = input_ids[0][-1].detach().cpu().item()
        token = self.tokenizer.decode(token_id)
        return any(char in token for char in self.stop_chars)


class ChatbotLocal:
    def __init__(self, base_model_dir, lora_adapter_dir, max_length, history_max_length, lora=True, use_chat_template=False):
        self.qa_history = ""
        self.max_length = max_length
        self.history_max_length = history_max_length
        self.use_chat_template = use_chat_template

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        if lora:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
            self.model.merge_and_unload()
            self.model.eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(base_model_dir)
            self.model.eval()

    def clear_history(self):
        self.qa_history = ""

    def update_history(self, question, answer):
        self.qa_history = f"{self.qa_history}我说：{question}\n你回答：{answer}\n"

    def truncate_history(self):
        lines = self.qa_history.splitlines()
        if len(lines) > 2:
            self.qa_history = '\n'.join(lines[2:])
            return False
        else:
            self.qa_history = ""
            return True

    def generate_response(self, user_input):
        while True:
            if self.use_chat_template:
                prompt = f"<|im_start|>system\n你是一个人工智能助手。<|im_end|>\n"
                prompt += f"<|im_start|>user\n对话历史：{self.qa_history}\n{user_input}<|im_end|>\n"
                prompt += f"<|im_start|>assistant\n"
            else:
                prompt = f"{user_input}\n回答："
            print(prompt)
            model_input = self.tokenizer([prompt], return_tensors='pt')
            if model_input["input_ids"].size(1) < self.max_length:
                break
            else:
                if self.truncate_history():
                    break
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, pad_token_id=self.tokenizer.eos_token_id)
        generatrion_kwargs = dict(**model_input,
                                  stopping_criteria=[CustomStoppingCriteria(["<|endoftext|>", "<|im_end|>"], self.tokenizer)],
                                  streamer=streamer,
                                  max_length=self.max_length, 
                                  max_new_tokens=None,
                                  pad_token_id=self.tokenizer.eos_token_id, 
                                  do_sample=True)

        thread = Thread(target=self.model.generate, kwargs=generatrion_kwargs)
        thread.start()
        chunks = []
        for chunk in streamer:
            chunks.append(chunk)
            yield chunk
        self.update_history(user_input, "".join(chunks))

    def chat(self):
        print("Welcome to the chat robot! Type \\quit to end the session, and \\newsession to start a new conversation.")
        while True:
            try:
                user_input = input("You: ")
                if user_input == "\\quit":
                    print("Goodbye!")
                    break
                elif user_input == "\\newsession":
                    self.clear_history()
                    print("Starting a new session.")
                else:
                    print("Assistant: ", end="", flush=True)
                    for text_chunk in self.generate_response(user_input):
                        print(text_chunk, end="", flush=True)
                    print()
                    self.clear_history()  # ad-hoc
            except KeyboardInterrupt:
                print("Interrupt")

if __name__ == "__main__":
    base_model_dir = "/workspace/nlp/Qwen2.5-3B"
    lora_adapter_dir = "/workspace/nlp/out/2025-01-17-02-43-47/checkpoint-10000"
    chatbot = ChatbotLocal(base_model_dir, lora_adapter_dir, 2048, 1000, False, False)
    chatbot.chat()