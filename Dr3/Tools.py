import os
import json
import time

sample_prefix = "./ToolPrompts/"
from collections import Counter
import openai
from abc import ABCMeta, abstractmethod

'''
    tool_name: 工具名称
    info： 工具信息，包括输入，输出
    prompt_name: prompt名字
    model_type：模型类型
    prefix：prompt前缀
'''


class BaseTool(metaclass=ABCMeta):

    def __init__(self, tool_name, info, prompt_name, model_type="openai", prefix="./ToolPrompts/"):
        super().__init__()
        self.tool_name = tool_name
        self.info = info

        path = os.path.join(prefix, prompt_name)
        with open(path, 'r') as f:
            lines = f.readlines()
        self.prompts = "".join(lines)
        self.model_type = model_type

        self.turn_message = Counter()
        self.total_message = Counter()

        self.success_call_num = 0
        self.total_total_call_num = 0
        self.point = 0
    def get_llm_response(self, inputs, stop=None):
        if stop is None:
            stop = ["None"]
        

        cnt = 8
        while cnt > 0:
            cnt -= 1
            try:
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=inputs,
                    temperature=0,
                    max_tokens=300,
                    top_p=0.001,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop,
                )
            except Exception as e:
                print(e)

        response_text = response["choices"][0]["text"]
        self.turn_message = response["usage"]
        self.total_message.update(self.turn_message)

        # record call result
        self.success_call_num += 1
        self.total_total_call_num += 1
        return response_text

    def reset(self):
        self.turn_message.clear()
        self.total_message.clear()
        self.success_call_num = 0
        self.total_total_call_num = 0

    '''
    自定义输入形式
    '''

    def construct_inputs(self, **kwargs):
        return ""

    '''
    对输出进行parse
    '''

    def parse_output(self, **kwargs):
        return ""

    def __call__(self, **kwargs):
        inputs = self.construct_inputs(**kwargs)
        new_inputs = inputs
        response = self.get_llm_response(new_inputs, kwargs.get("stop", None))
        return self.parse_output(response=response, **kwargs)

    def __repr__(self):
        return "name:" + self.tool_name + "\n" + "info:" + self.info + "\n"


import re


class RankerTools(BaseTool):
    def __init__(self, tool_name="ranker", info="Input: query, context, docs; Output: The most relevant doc",
                 prompt_name="ranker_prompts.txt", model_type="openai", prefix="./ToolPrompts/"):
        super().__init__(tool_name, info, prompt_name,
                         model_type,
                         prefix)

    def construct_inputs(self, **kwargs):
        query = kwargs["query"]  # str
        context = kwargs["context"]  # str
        docs = kwargs["docs"]  # lists
        inputs = '''Query:\n\t{}\nContext:\n\t{}\nDocuments:'''.format(query, context)
        for ix, d in enumerate(docs):
            inputs += f"\n\t[{ix + 1}] {d}"
        inputs += "\nThought:"

        return self.prompts + inputs

    def parse_output(self, response, **kwargs):
        try:
            res = re.search(r'Document \[(.*?)\]', response)
            return res.group(1)
        except Exception as e:
            try:
                res = re.search(r'\[(.*?)\]', response)
                return res.group(1)
            except Exception as e:
                return None


class StateQueryGenerationTools(BaseTool):
    def __init__(self, tool_name="state_generator", info="Input: last state ; Output: new action",
                 prompt_name="state_generator_prompts.txt", model_type="openai", prefix="./ToolPrompts/"):
        super().__init__(tool_name, info, prompt_name,
                         model_type,
                         prefix)

    def construct_inputs(self, **kwargs):
        question = kwargs["question"]  # str
        sub_questions = kwargs["sub_questions"]  # str
        evidences = kwargs["evidences"]  # lists
        # first_step
        if len(evidences) == 0:
            inputs = f"\nQuestion:{question}\nEvidence:No evidence.\n"
        else:
            inputs = f"\nQuestion:{question}\nEvidence:"
            evidence_str = ""
            for ix, x in enumerate(evidences):
                evidence_str += f"{ix + 1}. {x}"
            inputs += evidence_str + "\n"
            inputs += f"sub_questions:{sub_questions}\n"

        return self.prompts + inputs

    def parse_output(self, response, **kwargs):
        type = kwargs["return_type"]
        if type == "sub_question":
            try:
                res = re.search(r'Sub questions:(.+)', response)
                return res.group(1)
            except Exception as e:
                print(response)
                return None


class DecompsePromblemTools(BaseTool):
    def __init__(self, tool_name="decomposer", info="Input: Question ; Output: decompose questions",
                 prompt_name="decompose_prompts.txt", model_type="openai", prefix="./ToolPrompts/"):
        super().__init__(tool_name, info, prompt_name,
                         model_type,
                         prefix)

    def construct_inputs(self, **kwargs):
        question = kwargs["question"]  # str

        inputs = f"\nQuestion:{question}\n"

        return self.prompts + inputs

    def parse_output(self, response, **kwargs):
        try:
            thought = re.search("Thought:(.+)", response).group(1)
        except Exception as e:
            thought = "ohh.. thought can not be parse"

        try:
            decomposes = re.search("Decompose:(.+)", response).group(1)
        except Exception as e:
            decomposes = "ohh.. decompose can not be parse"
        return thought, decomposes


if __name__ == '__main__':
    pass
