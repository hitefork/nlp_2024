from dataclasses import dataclass
from typing import *

@dataclass
class ContentFactory:
    me: str = "小明"
    character: str = "凉宫春日"
    background: Optional[str] = "凉宫春日为北高一年五班的学生，生活在一个充满普通学生生活与超自然现象的校园中。她是 SOS 团（让世界变得更热闹的凉宫春日团）的团长，不知为何拥有能够改变世界能力的少女。拥有在无意识的情况下实现自己愿望的能力，虽然说只要她所希望的事，什么都能实现，但是可能受到春日「事实上那是不可能的」这种正常的思考方式的影响，她的想法好像不是全部能实现的样子。她原本一直过著平凡的生活，但自从在小学六年级时看了一场棒球赛后，便发现自己原来是十分渺小的，生活也是十分平凡乏味。因此，她在初中开始改变自己，不断做出各种怪事（如在晚上潜入学校操场画奇怪的符号），希望借此发现发现异常的事物，因此在东国中和北高都无人不晓她的名字，并将她当成怪人。到了北高后，因为阿虚的一句话而灵光乍现，成立 SOS 团。"
    personality: Optional[str] = "虽然成绩优异、容姿端丽、运动万能，但行为却非常地怪异，同时其性格极其冲动，一想到什么点子就做。好胜心极强，有着唯我独尊、旁若无人的性格。非常自以为是，超级任性，认为这个世界是以她为中心转动，脑子里永远不知道在想什么，并且情绪变化极端。"
    example_scene: Optional[str] = None
    example_qa_history: Optional[str] = "小明：「我到了！」\n凉宫春日：「好慢哦，要罚钱！」\n小明：「明明还没有九点。」\n凉宫春日：「就算没有迟到，最晚的人也要处罚，这就是规定！」"
    example_question: str = "我之前怎么没有听说？"
    example_answer: str = "因为是我刚刚才决定的啊，嘻嘻。"

    def get_system_content(self, use_example: bool = True):
        content = f"你现在扮演{self.character}，模仿{self.character}的性格。你使用{self.character}会使用的语气、方式和词汇进行对话续写。为此，你只需要输出一行 `{self.character}：「xxx」`，其中 xxx 是{self.character}会根据历史对话和场景作出的回复，注意，回复要符合{self.character}的性格，可长可短，有丰富性，但不超过 50 个字。\n"
        if self.background is not None:
            content += f"背景：{self.background}\n".format(background=self.background)
        if use_example:
            content += f"通过一个例子来解释规则：\n{self.get_example_content()}\n"
        return content
    
    def get_user_input_content(self, scene: Optional[str], qa_history: Optional[str], question: str):
        content = ""
        if self.personality is not None:
            content += f"{self.character}的性格：{self.personality}\n"
        if scene is not None:
            content += f"场景：{scene}\n"
        else:
            content += f"场景：现在{self.me}和{self.character}刚刚见面\n"
        if qa_history is not None:
            content += f"之前的对话：\n{qa_history}\n"
        content += f"{self.me}：「{question}」\n"
        content += f"现在{self.character}可能作出的回复是什么？"
        return content

    def get_example_content(self):
        content = f"比如，如果给出：\n\n```\n{self.get_user_input_content(self.example_scene, self.example_qa_history, self.example_question)}\n```\n\n" \
                 f"那么你应该输出：\n\n```\n{self.character}：「{self.example_answer}」\n```\n\n"
        return content
    
    def update_qa_history(self, qa_history: str, question: str, answer: str):
        qa_history += f"{self.me}：「{question}」\n{answer}\n"
        return qa_history
    
    def get_summarize_input_content(self, qa_history: str):
        return f"请你根据对话，总结成一段场景描述（比如{self.me}和谁在讨论什么话题、有什么关键信息等等）：{qa_history}"
    
    def get_assistant_prefix(self):
        return f"{self.character}：「"

class PromptFactory:
    def __init__(self, content_factory: ContentFactory):
        self.content_factory = content_factory

    def get_chat_prompt(self, scene: Optional[str], qa_history: Optional[str], user_input: str, use_example: bool = False):
        prompt = f"<|im_start|>system\n{self.content_factory.get_system_content(use_example)}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{self.content_factory.get_user_input_content(scene=scene, qa_history=qa_history, question=user_input)}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{self.content_factory.get_assistant_prefix()}"
        return prompt

    def get_summarize_prompt(self, qa_history: str):
        prompt = f"<|im_start|>system\n你是一个人工智能助手。<|im_end|>\n"
        prompt += f"<|im_start|>user\n{self.content_factory.get_summarize_input_content(qa_history)}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        return prompt
    
    def get_chat_prompt_messages(self, scene: Optional[str], qa_history: Optional[str], user_input: str, use_example: bool = False):
        messages = [
            {"role": "system", "content": self.content_factory.get_system_content(use_example)},
            {"role": "user", "content": self.content_factory.get_user_input_content(scene=scene, qa_history=qa_history, question=user_input)},
            {"role": "assistant", "content": self.content_factory.get_assistant_prefix(), "prefix": True}
        ]
        return messages

    def get_summarize_prompt_messages(self, qa_history: str):
        messages = [
            {"role": "system", "content": "你是一个人工智能助手。"},
            {"role": "user", "content": self.content_factory.get_summarize_input_content(qa_history)}
        ]
        return messages