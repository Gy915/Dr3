# -*- coding: utf-8 -*-
import os
import pprint
import sys
import traceback
os.environ["OPENAI_API_KEY"] = "" # your api key here


openai.api_key = os.environ["OPENAI_API_KEY"]
import requests

debug = False if sys.platform.lower() != 'Darwin'.lower() else True
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)
# 方便引用react里的包
# if debug:
#     react_path = '/Users/caoyuanbin/gitProject/ReAct/'
# else:
#     react_path = '/home/lingzun.cyb/MyReAct/'
# # 使用自带prompt
# if debug:
#     prompt_folder = '/Users/caoyuanbin/gitProject/ReAct/prompts/'
# else:
#     prompt_folder = '/home/lingzun.cyb/MyReAct/prompts/'
#
# sys.path.append(react_path)
sys.path.append('/Users/janbei/PycharmProjects/langchain')
sys.path.append('/mnt/nlp_nas_milvus/gaoy/project/langchain')
sys.path.append('/mnt/nlp_nas_milvus/gaoy/project/langchain/PE')

shot_num = 6
# 设置日志格式
log_prefix = "/Users/janbei/PycharmProjects/langchain/PE/scenario/react_exp/exp_logs_2"
begin = 100
end = 120
log_name = f"2wiki_begin_{begin}_end_{end}_shot_{shot_num}"

prompt_folder = "../prompts/ours/"


# python PE/scenario/react_exp/run.py 3 0
# python PE/scenario/react_exp/run.py 3 1
# python PE/scenario/react_exp/run.py 3 2
total_rank = int(sys.argv[1])
local_rank = int(sys.argv[2])


llm_type = sys.argv[3]
search_strategy = sys.argv[4]
if len(sys.argv) >= 6:
    file_name = sys.argv[5]
else:
    file_name = "temp"
llm_call_num = 0
proxy = False
import time

if llm_type == "openai":
    # 使用OpenAI
    import openai


    def llm(prompt, stop=["None"]):
        if proxy:
            chatgpt = Sample.main(prompt)
            return chatgpt
        else:
            cnt = 0
            while cnt < 3:
                try:
                    response = openai.Completion.create(
                        model="text-davinci-002",
                        prompt=prompt,
                        temperature=0,
                        top_p=0.001,
                        max_tokens=200,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop,
                        # timeout=3600,
                    )
                    break
                except Exception as e:
                    time.sleep(0.5)
                    cnt += 1

            return response["choices"][0]["text"]

if llm_type == "cainiao":
    def llm(prompt, stop=["\n"]):
        global llm_call_num
        response = llm_pxy(prompt=prompt, stop=stop)
        return response

print("test llm:", llm("hello", stop=["None"]))
llm_pxy.reset_token()
print()


def step(env, **inputs):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(**inputs)
        except requests.exceptions.Timeout:
            attempts += 1


import json

# prompt_file = 'prompts_naive.json'
# with open(prompt_folder + prompt_file, 'r') as f:
#     prompt_dict = json.load(f)
#
# webthink_prompt = prompt_dict['webthink_simple6']

# if search_strategy == "baseline+graph":
#     with open("./graph_search_prompt.json", 'r') as f:
#         webthink_prompt = json.load(f)

#
# instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
# (1) Search[query], which searches query and try to answer the query if possible, and returns the first paragraph if it exists. If not, it will return the similar resullt.
# (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
# (3) Finish[answer], which returns the answer and finishes the task.
# Here are some examples.
# """
# (2) Handle[query], which turn the query to SPARQL language and return the execute result if it exits. If not, it will return some possible results.
# webthink_prompt = instruction + webthink_prompt


if search_strategy in ["colbert", "colbert+ranker", "colbert+ranker+askteacher"]:
    prompt_name = "colbert.json"

if search_strategy in ["colbert+actor"]:
    prompt_name = f"colbert_actor_{shot_num}_shot.json"


with open(prompt_folder + prompt_name) as f:
    prompt_dict = json.load(f)

instruct = prompt_dict["instructs"]
samples = prompt_dict["samples"]
prompts = instruct + samples
print("prompt:")
print(prompts)

print("\n\n")

import re


def react_step(prompt, question, to_print=True):
    # 第一步decompose特殊处理
    total_steps = ""
    decompose_task_action = llm(prompt + "Decomposition:", stop=[f"\nObservation 1:"])
    i = 1
    decompose_task_action = decompose_task_action.strip("\n| ")
    res = re.match(f"(.*)Task {i}: (.*)Action {i}: (.*)", decompose_task_action, re.DOTALL)
    decompose = res.group(1).strip("\n| ")
    task = res.group(2).strip("\n| ")
    action = res.group(3).strip("\n| ")
    inputs = {"action": action[0].lower() + action[1:], "task": task, "question": question}
    obs, r, done, info = step(env, **inputs)
    obs = obs.replace('\\n', '')
    step_str = f"Decomposition: {decompose}\nTask {i}: {task}\nAction {i}: {action}\nObservation {i}: {obs}\n"
    total_steps += step_str
    if to_print:
        print(step_str)
    prompt += step_str
    final_conclusion = ""
    composition = ""
    for i in range(2, 8):
        res = llm(prompt + f"Conclusion {i - 1}:", stop=[f"\nObservation {i}:"])
        res = res.strip("\n| ")
        if "Composition:" in res:  # compose 特殊处理
            conclude_compose_finish = res
            try:
                res = re.match(f"(.*)Composition: (.*)Finish: (.*)", conclude_compose_finish, re.DOTALL)
                last_conclusion = res.group(1).strip("\n| ")
                composition = res.group(2).strip("\n| ")
                answer = res.group(3).strip("\n| ")[1:-1]  # []
                step_str = f"Conclusion {i - 1}: {last_conclusion}\nComposition: {composition}\nFinish:[{answer}]"
                final_conclusion = f"Conclusion {i - 1}: {last_conclusion}\n"
                total_steps += step_str
                obs, r, done, info = step(env, action=f"finish[{answer}]")

            except Exception as e:
                pass


        else:  # 正常的task
            try:
                re_res = re.match(f"(.*)Task {i}: (.*)Action {i}: (.*)", res, re.DOTALL)
                last_conclusion = re_res.group(1).strip("\n| ")
                # todo: task-action-obs-conclusion -> add teacher
                next_task = re_res.group(2).strip("\n| ")
                action = re_res.group(3).strip("\n| ")
            except Exception as e:
                print("=" * 10 + f"Can not divide LLM output to Conclusion task action, try to regard \"{res}\" as "
                                 f"conclude" + "=" * 10)
                last_conclusion = res
                task_action = llm(prompt + f"Conclusion {i - 1}: {res}\nTask {i}:", stop=[f"\nObservation {i}:"])
                res = re.match(f"(.*)Action {i}: (.*)", task_action, re.DOTALL)
                next_task = res.group(1).strip("\n| ")
                action = res.group(2).strip("\n| ")
                # 错误处理

            inputs = {"action": action[0].lower() + action[1:], "task": task, "question": question}
            obs, r, done, info = step(env, **inputs)
            obs = obs.replace('\\n', '')
            step_str = f"Conclusion {i - 1}: {last_conclusion}\nTask {i}: {next_task}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            total_steps += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        answer = llm(prompt + "\nFinish: ")
        answer = answer.strip("\n| ")[1:-1]
        obs, r, done, info = step(env, action=f"finish[{answer}]")
        total_steps += f"Composition:\nFinish:{answer}"

    return info, prompt + final_conclusion, "Composition: " + composition, total_steps,


from collections import defaultdict


def react_2(idx=None, prompt=prompts, to_print=True, save_process=True):
    records = defaultdict(str)
    print_strs = ""
    question = env.reset(idx=idx)

    if to_print:
        print(idx, question)
        print()
        print_strs += f"{idx} {question}\n"
    prompt += question + "\n"
    compose_teacher_tool = JudgeTeacherTool(prefix="../teachers/prompts/", prompt_name="compose_teacher_3.txt")
    info, prompt_before_compose, composition, total_steps = react_step(prompt, question, to_print=to_print)
    pred_answer = info["answer"]
    em = info["em"]
    cover_em = info["cover_em"]
    f1 = info["f1"]
    gold_answer = info["gt_answer"]
    if save_process:
        records["idx"] = idx
        records["question"] = question
        records["steps"] = total_steps
        records["stu_answer"] = pred_answer
        records["stu_em"] = em
        records["stu_cover_em"] = cover_em
        records["f1"] = f1
        records["gold_answer"] = gold_answer
    jud, type = compose_teacher_tool(question=question, answer=pred_answer, get_response = True)
    print(f"========Teacher Judgement: {jud}, Teacher Judge Type:{type}===========")
    records["teacher_jud"] = jud
    records["teacher_type"] = type
    after_teacher_answer = pred_answer
    # 类型不对，加上type后重新输出
    if jud == "NO":

        step_prompt = "Composition: " + f"The answer is not \"{pred_answer}\", and "
        # step_prompt = f"Composition: I need to know that The answer is not \"{pred_answer}\", and "
        if to_print:
            print("After Teacher Prompt: ", step_prompt)
            print("-" * 5)
        prompt_new = prompt_before_compose + step_prompt
        # prompt_new = prompt_before_compose + f"\nComposition: {composition}" + f"\nOh I'm sorry, the answer is not \"{pred_answer}\", because the type of {pred_answer} is wrong, so " #后处理
        response = llm(prompt_new, stop=["None"])
        print("After Teacher Response: ", response)
        print("-" * 5)

        try:
            re_res = re.search(f"(.*)Finish: (.*)", response, re.DOTALL)
            pred_answer = re_res.group(2)[1:-1]  # [answer]
        except Exception as e:
            response = llm(prompt_new + response.strip("\n| ") + "\nFinish: ")
            pred_answer = response.strip("\n| ")[1:-1]
        print("After Teacher Pred: ", pred_answer)
        print("-" * 5)

        after_teacher_answer = pred_answer
        _, _, _, info = step(env, action=f"finish[{after_teacher_answer}]")
    records["teacher_answer"] = after_teacher_answer

    if to_print:
        print(info)
    return info, records


# import random
import time

def react_3(idx=None, prompt=prompts, to_print=True, save_process=True):
    records = defaultdict(str)
    print_strs = ""
    question = env.reset(idx=idx)

    if to_print:
        print(idx, question)
        print()
        print_strs += f"{idx} {question}\n"
    prompt += question + "\n"
    compose_teacher_tool = JudgeTeacherTool(prefix="../teachers/prompts/", prompt_name="compose_teacher_3.txt")
    info, prompt_before_compose, composition, total_steps = react_step(prompt, question, to_print=to_print)
    pred_answer = info["answer"]
    em = info["em"]
    cover_em = info["cover_em"]
    f1 = info["f1"]
    gold_answer = info["gt_answer"]
    if save_process:
        records["idx"] = idx
        records["question"] = question
        records["steps"] = total_steps
        records["stu_answer"] = pred_answer
        records["stu_em"] = em
        records["stu_cover_em"] = cover_em
        records["f1"] = f1
        records["gold_answer"] = gold_answer
    jud, type = compose_teacher_tool(question=question, answer=pred_answer, get_response = True)
    print(f"========Teacher Judgement: {jud}, Teacher Judge Type:{type}===========")
    records["teacher_jud"] = jud
    records["teacher_type"] = type

    if to_print:
        print(info)
    return info, records
# idxs = list(range(7405))
# random.Random(233).shuffle(idxs)
def main(env, log_name, begin, end):
    ems = []
    cover_ems = []
    infos = []
    old_time = time.time()

    # given_idx = list(range(len(env.env.data)))[200:230]
    # given_idx = list(range(10))
    # given_idx = list(range(len(env.data)))
    total_records = {"prompts": prompts, "records": []}
    last_total_token = 0
    cnt = 0
    for i in range(begin, end):

        cnt += 1
        print('-----start------')
        print('QUESTION: ' + '\t'.join(env.data[i]) + '\n')
        try:
            info, records = react_3(i, to_print=True)
            total_records["records"].append(records)
            # print(records)
            ems.append(info['em'])
            cover_ems.append(info["cover_em"])
            infos.append(info)
            print(f"em:{sum(ems)} / {len(ems)}, cover_em:{sum(cover_ems)} / {len(cover_ems)}")
            # env.write()

        except Exception as e:
            traceback.print_exc()
            continue
        # this_question_token = llm_pxy.use_token_message["totalTokens"] - last_total_token
        # last_total_token = llm_pxy.use_token_message["totalTokens"]

        # print(f"this question cost token: {this_question_token}\n")
        print('-----end------')
        logging.info(f"idx:{i} finished, em:{sum(ems)} / {len(ems)}, cover_em:{sum(cover_ems)} / {len(cover_ems)}")
        print("\n\n")
    with open(f"{log_prefix}/{log_name}.json", 'w') as f:
        json.dump(total_records, f, indent=2)
    # print("avg total", llm_pxy.use_token_message["totalTokens"] / len(end-begin + 1))


if __name__ == '__main__':
    sys.stdout = open(f'{log_prefix}/{log_name}.log', 'w')
    import wikienv_2, wrappers_2

    print("prompt:")
    print(prompts)
    env = wikienv_2.WikiEnv(search_strategy=search_strategy, model_type=llm_type)
    env = wrappers_2.HotPotQAWrapper(env, split="dev", local_rank=local_rank, total_rank=total_rank)
    main(env, log_name, begin=begin, end=end)
    sys.stdout.close()
