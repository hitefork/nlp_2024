from threading import Thread
from typing import *
import json

from prompt_factory import ContentFactory, PromptFactory

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria
from peft import PeftModel
from openai import OpenAI


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_chars, tokenizer):
        self.tokenizer = tokenizer
        self.stop_chars = stop_chars

    def __call__(self, input_ids, scores, **kwargs):
        token_id = input_ids[0][-1].detach().cpu().item()
        token = self.tokenizer.decode(token_id)
        return any(char in token for char in self.stop_chars)

class ChatbotLocal:
    def __init__(self, base_model_dir, lora_adapter_dir, max_length):
        self.qa_history = ""
        self.max_length = max_length
        self.content_factory = ContentFactory()
        self.prompt_factory = PromptFactory(self.content_factory)

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
        self.model.merge_and_unload()
        self.model.eval()

    def clear_history(self):
        self.qa_history = ""

    def update_history(self, question, answer):
        self.qa_history = self.content_factory.update_qa_history(self.qa_history, question, answer)

    def summarize_history(self) -> str:
        prompt = self.prompt_factory.get_summarize_prompt(self.qa_history)
        model_input = self.tokenizer([prompt], return_tensors='pt')
        self.qa_history = self.tokenizer.batch_decode(
            self.model.generate(**model_input, 
                                max_length=self.max_length, 
                                max_new_tokens=None,
                                pad_token_id=self.tokenizer.eos_token_id, 
                                do_sample=True)[:,model_input["input_ids"].size(1):], 
            skip_special_tokens=True)[0]
        return self.qa_history

    def generate_response(self, user_input):
        for strategy in ["summarize", "clear", "raise"]:
            prompt = self.prompt_factory.get_chat_prompt(scene=None, qa_history=self.qa_history, user_input=user_input)
            model_input = self.tokenizer([prompt], return_tensors='pt')
            if model_input["input_ids"].size(1) < self.max_length:
                break
            else:
                if strategy == "summarize":
                    self.summarize_history()
                if strategy == "clear":
                    self.clear_history()
                if strategy == "raise":
                    raise Exception
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, pad_token_id=self.tokenizer.eos_token_id)
        generatrion_kwargs = dict(**model_input,
                                  stopping_criteria=[CustomStoppingCriteria(["<|endoftext|>", "<|im_end|>", "」"], self.tokenizer)],
                                  streamer=streamer,
                                  max_length=self.max_length, 
                                  max_new_tokens=None,
                                  pad_token_id=self.tokenizer.eos_token_id, 
                                  do_sample=True)

        thread = Thread(target=self.model.generate, kwargs=generatrion_kwargs)
        thread.start()
        chunks = [self.content_factory.get_assistant_prefix()]
        yield chunks[0]
        for chunk in streamer:
            chunks.append(chunk)
            yield chunk
        self.update_history(user_input, "".join(chunks))

    def chat(self):
        print("Welcome to the chat robot! Type \\quit to end the session, and \\newsession to start a new conversation.")
        while True:
            user_input = input("You: ")
            if user_input == "\\quit":
                print("Goodbye!")
                break
            elif user_input == "\\newsession":
                self.clear_history()
                print("Starting a new session.")
            elif user_input == "\\summarize":
                print(self.summarize_history())
            else:
                print("Assistant: ", end="", flush=True)
                for text_chunk in self.generate_response(user_input):
                    print(text_chunk, end="", flush=True)
                print()


class ChatbotAPI:
    def __init__(self, max_length, api_key, base_url, model):
        self.qa_history = ""
        self.max_length = max_length
        self.content_factory = ContentFactory()
        self.prompt_factory = PromptFactory(self.content_factory)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def clear_history(self):
        self.qa_history = ""

    def update_history(self, question, answer):
        self.qa_history = self.content_factory.update_qa_history(self.qa_history, question, answer)

    def summarize_history(self) -> str:
        messages = self.prompt_factory.get_summarize_prompt_messages(self.qa_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        self.qa_history = response.choices[0].message.content
        return self.qa_history

    def generate_response(self, user_input):
        for strategy in ["summarize", "clear", "raise"]:
            messages = self.prompt_factory.get_chat_prompt_messages(scene=None, qa_history=self.qa_history, user_input=user_input, use_example=True)
            
            if sum(len(message["content"]) for message in messages) < self.max_length:
                break
            else:
                if strategy == "summarize":
                    self.summarize_history()
                if strategy == "clear":
                    self.clear_history()
                if strategy == "raise":
                    raise Exception
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=["」"],
            stream=True
        )
        chunks = [self.content_factory.get_assistant_prefix()]
        yield chunks[0]
        for chunk in response:
            chunks.append(chunk.choices[0].delta.content)
            yield chunks[-1]
        chunks.append("」")
        yield chunks[-1]
        self.update_history(user_input, "".join(chunks))

    def chat(self):
        print("Welcome to the chat robot! Type \\quit to end the session, and \\newsession to start a new conversation.")
        while True:
            user_input = input("You: ")
            if user_input == "\\quit":
                print("Goodbye!")
                break
            elif user_input == "\\newsession":
                self.clear_history()
                print("Starting a new session.")
            elif user_input == "\\summarize":
                print(self.summarize_history())
            else:
                print("Assistant: ", end="", flush=True)
                for text_chunk in self.generate_response(user_input):
                    print(text_chunk, end="", flush=True)
                print()


def modify_content_factory(chatbot: Union[ChatbotAPI, ChatbotLocal], json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    chatbot.content_factory = ContentFactory(**data)
    chatbot.prompt_factory = PromptFactory(chatbot.content_factory)


if __name__ == "__main__":
    base_model_dir = "/workspace/nlp/Qwen2.5-3B"
    lora_adapter_dir = "/workspace/nlp/out/2025-01-17-02-43-47/checkpoint-10000"
    # 可以选择使用本地模型还是 API
    # chatbot = ChatbotLocal(base_model_dir, lora_adapter_dir, 2048)
    chatbot = ChatbotAPI(10000, "API_KEY", "https://api.deepseek.com/beta", "deepseek-chat")
    chatbot.chat()