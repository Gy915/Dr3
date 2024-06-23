import ast
import copy
import json
import logging
import time
import gym
import requests
from bs4 import BeautifulSoup
from colbert_search_engine import ColBERTv2
import openai
from collections import Counter
from util import openai_llm, cainiao_llm, Colored
import re
from metrics import get_metrics
from judge_teacher import JudgeTeacherTool
from re_decompose_teacher import ReDecomposeTeacher
from cot_judger import cot_judger

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, str)


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class CorrectMethods:
    TRY_NUMS = 10

    def __init__(self,
                 prompt,

                 ):
        super().__init__()
        self.page = None  # current Wikipedia page
        self.obs = None  # current observation
        self.lookup_keyword = None  # current lookup keyword
        self.lookup_list = None  # list of paragraphs containing current lookup keyword
        self.lookup_cnt = None  # current lookup index
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        self.query_to_entity = None
        self.prompts = prompt
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.success_call_num = 0
        self.total_total_call_num = 0
        self.turn_message = Counter()
        self.total_message = Counter()
        self.success_call_num = 0
        self.total_total_call_num = 0

        # set search strategy

    def setup(self, search_strategy="colbert", correct_type="dfs_passage", model_type="openai", ):
        self.search_strategy_map(search_strategy)
        self.correct_strategy_map(correct_type)

        if model_type == "openai":
            self.llm = openai_llm
        elif model_type == "cainiao":
            self.llm = cainiao_llm

    def parse_memory(self, question, memory, stu_answer, ):
        splits = memory.split("\n")
        valid = True
        try:
            assert splits[0].startswith("Decomposition:")
            assert splits[-1].startswith("Finish:")
            assert splits[-2].startswith("Composition:")
        except Exception as e:
            valid = False
            return valid, None
        tasks = splits[1:-2]
        split_tasks = []
        i = 0
        while 1:
            try:
                split_tasks.append((tasks[i], tasks[i + 1], tasks[i + 2], tasks[i + 3]))
                assert tasks[i].startswith("Task")
                assert tasks[i + 1].startswith("Action")
                assert tasks[i + 2].startswith("Observation")
                assert tasks[i + 3].startswith("Conclusion")
            except Exception as e:
                valid = False
                return valid, None

            i += 4
            if i >= len(tasks):
                break
        res = {}
        res["question"] = question
        res["decomposition"] = splits[0]
        res["tasks"] = split_tasks
        res["composition"] = splits[-2]
        res["stu_answer"] = stu_answer
        res["finish"] = splits[-1]

        return valid, res

    def correct_strategy_map(self, correct_strategy):
        if correct_strategy not in ['dfs_passage', "re_decompose", "re_compose", "total", ]:
            print("invalid correct strategy")
            return False
        if correct_strategy == "dfs_passage":
            self.correct_strategy = self.colbert_dfs_strategy
        if correct_strategy == "re_decompose":
            self.correct_strategy = self.re_decompose_strategy
        if correct_strategy == "re_compose":
            self.correct_strategy = self.re_compose_strategy
        if correct_strategy == "total":
            self.correct_strategy = self.total_strategy
        return True

    def search_strategy_map(self, search_strategy, **kwargs):
        if search_strategy not in ['wiki', 'colbert']:
            print("invalid search strategy")
            return False

        if search_strategy == "wiki":
            self.search_strategy = self.wiki_strategy
        if search_strategy == "colbert":
            self.colbert_engine = ColBERTv2()
            self.search_strategy = self.colbert_strategy

        return True

    def correct(self, question, memory, stu_answer, **kwargs):
        pred, prompt_final = self.correct_strategy(question, memory, stu_answer, **kwargs)
        if kwargs.get("return_task_prompt", False):
            if prompt_final is None:
                return pred, None
            else:
                return pred, prompt_final[len(self.prompts):]  # prompt
        else:
            return pred


    '''
    三个纠错综合，
    teacher_tool: Discriminator
    '''
    def total_strategy(self, question, memory, stu_answer, teacher_tool, re_decompose_tool, **kwargs):
        color = Colored()
        if question.startswith("Question"):
            question_wo_prefix = question[len("Question: "):]
        else:
            question_wo_prefix = question
        question_w_prefix = "Question: " + question
        final_answer = stu_answer
        i = 0
        steps_answers = {}
        try:
            answer, prompt_final = self.re_compose_strategy(question_w_prefix, memory, stu_answer, **kwargs)
        except Exception as e:
            print(color.red(f"some error in recompose strategy....\n{e}"))
            return final_answer, None
        else:
            print(f"The {i} step answer is {answer}")
            steps_answers[i] = answer
            i += 1

        final_answer = answer
        jud, res = teacher_tool(question=question_wo_prefix, answer=final_answer, )
        if jud != "YES":
            try:
                answer, prompt_final = self.colbert_dfs_strategy(question_w_prefix, memory, stu_answer, teacher_tool,
                                                                 **kwargs)
            except Exception as e:
                print(color.red(f"some error in dfs strategy....\n{e}"))
                return final_answer, steps_answers
            else:
                print(f"The {i} step answer is {answer}")
                steps_answers[i] = answer
                i += 1
            final_answer = answer
            jud, res = teacher_tool(question=question_wo_prefix, answer=final_answer, )
            if jud != "YES":
                try:
                    answer, prompt_final = self.re_decompose_strategy(question_w_prefix, memory, stu_answer,
                                                                      re_decompose_tool, **kwargs)
                except Exception as e:
                    print(color.red(f"some error in re-decompose strategy....\n{e}"))
                    return final_answer, steps_answers
                else:
                    print(f"The {i} step answer is {answer}")
                    steps_answers[i] = answer
                    i += 1
                final_answer = answer

        return final_answer, prompt_final


    '''
    Re-solve
    '''
    def colbert_dfs_strategy(self, question, memory, stu_answer, teacher_tool, **kwargs):
        # 使用colbert结果dfs搜索，获取矫正结果

        if question.startswith("Question"):
            question_wo_prefix = question[len("Question: "):]
        else:
            question_wo_prefix = question
        question_w_prefix = "Question: " + question

        def get_cumulate_prompt(prompts, parsed_memory, task_id=-1):
            '''
            返回[0, task_id)个task对应的prompt
            '''
            if task_id >= len(parsed_memory["tasks"]) or task_id == -1:
                task_id = len(parsed_memory["tasks"])
            prompt = prompts + \
                     question_w_prefix + "\n" + \
                     parsed_memory["decomposition"]
            for i in range(task_id):
                prompt += "\n" + "\n".join(parsed_memory["tasks"][i])
            return prompt  # Task1:xxx, Action1: xxx, Observation: xxx, Conclude:xxx...

        color = Colored()
        if kwargs.get("to_print", False):
            print("*" * 10 + "ORIGINAL" + "*" * 10)
            print(memory)
        valid, parsed_memory = self.parse_memory(question_w_prefix, memory, stu_answer, )
        if valid is False:
            return stu_answer, None
        max_num = kwargs.get("search_depth", 3)
        total_task_num = len(parsed_memory["tasks"])
        for num in range(total_task_num - 1, -1, -1):
            idx = num + 1
            prompt = get_cumulate_prompt(self.prompts, parsed_memory, num)  # 0-num-1
            # colbert 结果进行替换
            for default_idx in range(2, max_num + 1):
                current_task = parsed_memory["tasks"][num][0]
                prompt += "\n" + current_task
                current_action = parsed_memory["tasks"][num][1]
                prompt += "\n" + current_action
                query = re.search("(.*)\[(.*)\]", current_action)
                query = query.group(2)
                new_obs = self.get_obs(query, idx, rank=default_idx)
                prompt += new_obs
                pred, prompt_final = self.base_react(prompt=prompt, idx=idx)
                jud, response = teacher_tool(question=question_wo_prefix,
                                             answer=pred,
                                             get_response=True)

                dfs_res = prompt_final

                # print('+' * 10 + f"Task:{idx}, Colbert Result:{default_idx}" + '+' * 10)
                # print("Pred Answer: ", pred)
                # print("Judgement: ", response)
                # print(prompt_final[len(self.prompts):])
                # print(color.red("TEACHER RESPONSE"))
                # print(response.strip())
                # print("\n")
                if jud == "YES":
                    return pred, dfs_res
        return stu_answer, dfs_res


    '''
    Re-Decompose
    '''
    def re_decompose_strategy(self, question, memory, stu_answer, re_decompose_tool, **kwargs):
        # question: Question:xxxx
        # memory: Decomposition: xxx, Task 1:xxx,
        color = Colored()
        if kwargs.get("to_print", False):
            print("*" * 10 + "ORIGINAL" + "*" * 10)
            print(memory)
        valid, parsed_memory = self.parse_memory(question, memory, stu_answer, )
        if valid is False:
            return stu_answer, None
        question = parsed_memory["question"][len("Question:"):].strip()
        decompose = parsed_memory["decomposition"][len("Decomposition:"):].strip()
        new_compose, thought = re_decompose_tool(question=question, decomposition=decompose, **kwargs)
        prompt = self.prompts + f"Question: {question}" + f"\nDecomposition: {new_compose}"
        input_params = kwargs
        input_params["prompt"] = prompt
        pred_answer, prompt_final = self.base_react(**input_params)
        return pred_answer, prompt_final



    '''
    Re-Compose
    '''
    def re_compose_strategy(self, question, memory, stu_answer, **kwargs):
        color = Colored()
        if kwargs.get("to_print", False):
            print("*" * 10 + "ORIGINAL" + "*" * 10)
            print(memory)
        valid, parsed_memory = self.parse_memory(question, memory, stu_answer, )
        if valid is False:
            return stu_answer, None
        prompt = self.prompts + \
                 question + "\n" + \
                 parsed_memory["decomposition"]
        for i in range(len(parsed_memory["tasks"])):
            prompt += "\n" + "\n".join(parsed_memory["tasks"][i])
        prompt += f"\nComposition: The answer is not \"{stu_answer}\", and "
        res = self.base_react_step(prompt, re_composition=True)

        # 获得Task的Prompt
        task_prompt = res["prompt"][len(self.prompts) + 1:]
        if kwargs.get("to_print", False):
            print(task_prompt)
        answer = res.get("answer", stu_answer)
        return answer, res["prompt"]

    @staticmethod
    def get_page_obs(page):
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])  # 只返回了前五？

    def wiki_strategy(self, entity, **kwargs):
        """
            entity: str
            如果存在实体，返回搜索结果；否则，返回5个相似的结果
        """
        entity_ = entity.replace(" ", "+")
        search_url = WIKI_PROXY_ADDR.replace('@@entity@@', entity_)
        old_time = time.time()
        # requests_cache.install_cache('wiki_cache')  # 加个缓存，需要pip install requests_cache
        response_text = requests.get(search_url).text
        self.search_time += time.time() - old_time
        soup = BeautifulSoup(response_text, features="html.parser")  # html处理
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            # logging.info(f"No Search Result For [{entity}], Try Graph Search....")
            obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."

        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.wiki_strategy("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                obs = self.get_page_obs(self.page)
        return obs

    def colbert_strategy(self, query, **kwargs):
        res = self.colbert_engine(query, kwargs.get("rank", 1))  # [p1, p2, ...p_index]
        obs = res[kwargs.get("rank", 1) - 1]
        return obs

    # ReAct基本框架，需要给出prompt
    def base_react(self, **kwargs):
        input_params = kwargs
        pred_answer = None
        flag = False
        prompt_final = input_params["prompt"]
        for i in range(self.TRY_NUMS):
            res = self.base_react_step(**input_params)
            prompt_final = res["prompt"]
            if res.get("is_final_step", False):
                pred_answer = res["answer"]
                flag = True
                break
            else:
                input_params = res
        if flag is False:  # 没有结果，强行结束
            prompt_final += "\nComposition:"
            input_params["prompt"] = prompt_final
            res = self.base_react_step(re_composition=True, **input_params)
            try:
                prompt_final = res["prompt"]
                pred_answer = res["answer"]
            except Exception as e:
                return "none", prompt_final
        return pred_answer, prompt_final

    def base_react_step(self, prompt, **kwargs):
        color = Colored()
        '''
        方便每一步重做，从目前状态开始，完成ReAct下一步
        '''
        res = {}
        last_state = prompt.split("\n")[-1]
        if last_state.startswith("Question:"):
            # Re-Decomposition
            response = self.llm(prompt + "\nDecomposition: ", stop=["\nObservation 1"])
            response = response.strip("\n| ")
            i = 1
            try:
                res_new = re.match(f"(.*)Task {i}: (.*)Action {i}: (.*)", response, re.DOTALL)
                decompose = res_new.group(1).strip("\n| ")
                task = res_new.group(2).strip("\n| ")
                action = res_new.group(3).strip("\n| ")
            except Exception as e:  # 只有decompose
                response = response.strip("\n| ")
                decompose = response
                task_action = self.llm(prompt + f"\nDecomposition: {decompose}\nTask {i}: ", stop=["\nObservation 1"])
                task_action_new = re.match(f"(.*)Action {i}: (.*)", task_action, re.DOTALL)
                task = task_action_new.group(1).strip("\n| ")
                action = task_action_new.group(2).strip("\n| ")
            step_str = f"\nDecomposition: {decompose}\nTask 1: {task}\nAction 1: {action}"
            prompt += step_str
            res["idx"] = 1
            res["query"] = action[len("Search["):-1].strip("\n| ")
            res["prompt"] = prompt
            res["is_final_step"] = False

        if last_state.startswith("Decomposition"):
            # Begin_Task1
            response = self.llm(prompt + "\nTask 1:", stop=["\nObservation 1"])
            response = response.strip("\n| ")
            try:
                res_new = re.match(f"(.*)Action 1: (.*)", response, re.DOTALL)
                task = res_new.group(1).strip("\n| ")
                query = res_new.group(2)[len("Search["):-1].strip("\n| ")
            except Exception as e:
                # res = res.s
                # res = self.llm(prompt + f"\nTask 1: {res}\n")
                print(color.red(
                    f"In Decomposition stage, response\n \"{response}\" \ncan not parse to task 1, action 1 format"))
                pass
            step_str = f"\nTask 1: {task}\nAction 1:[{query}]"
            prompt += step_str
            res["idx"] = 1
            res["query"] = query
            res["prompt"] = prompt
            res["is_final_step"] = False

        if last_state.startswith("Action"):
            # query
            idx = kwargs["idx"]
            query = kwargs["query"]
            obs = self.get_obs(query, idx)
            step_str = obs
            prompt += obs
            res["prompt"] = prompt
            res["idx"] = kwargs["idx"]
            res["is_final_step"] = False

        if last_state.startswith("Observation"):
            # Conclude and Task
            idx = kwargs["idx"]
            step_str, inform = self.get_next_conclude_task_or_answer_generate(prompt, idx)
            res = copy.deepcopy(inform)
            prompt += step_str
            res["prompt"] = prompt
            res["idx"] = idx + 1

        if last_state.startswith("Composition"):  # Composition: xxxx
            # New Finish
            if kwargs.get("re_composition", False):
                new_res = self.llm(prompt)
                try:
                    temp = re.match(f"(.*)Finish: (.*)", new_res, re.DOTALL)
                    compose = temp.group(1).strip("\n| ")
                    answer = temp.group(2).strip("\n| ")  # [answer]
                    step_str = compose + f"\nFinish: {answer}"
                    prompt += step_str
                    res["compose"] = compose
                    res["answer"] = answer[1:-1]
                    res["prompt"] = prompt

                except Exception as e:
                    compose = new_res.strip("\n| ")
                    answer = self.llm(prompt + f"\nComposition:{compose}\nFinish:")
                    answer = answer.strip("\n| ")[1:-1]
                    step_str = compose + f"\nFinish: [{answer}]"
                    prompt += step_str
                    res["compose"] = compose
                    res["answer"] = answer
                    res["prompt"] = prompt


            else:  # 直接重新生成answer
                new_res = self.llm(prompt + "\nFinish:")
                answer = new_res.strip("\n| ")[1:-1]
                step_str = f"\nFinish: [{answer}]"
                prompt += step_str
                res["answer"] = answer
                res["prompt"] = prompt
            res["is_final_step"] = True

        if last_state.startswith("Finish"):
            # end, gold_answer
            answer = last_state[len('Finish: ['):-1]
            res["prompt"] = prompt
            res["answer"] = answer
            res["is_final_step"] = False
        if kwargs.get("to_print", False):
            print(step_str)
        res["to_print"] = kwargs.get("to_print", False)
        return dict(res)

    def get_obs(self, query, current_idx, **kwargs):
        step_str = f"\nObservation {current_idx}: "
        obs = self.search_strategy(query, **kwargs)
        obs = obs.replace('\\n', '')
        step_str += obs
        return step_str

    def get_next_conclude_task_or_answer_generate(self, prompt, current_idx):
        # prompt: samples+ decompose + (task + action + observation)_N
        is_final_step = False
        step_str = ""
        res = self.llm(prompt + f"\nConclusion {current_idx}:", stop=[f"\nObservation {current_idx + 1}:"])
        res = res.strip("\n| ")
        inform = {}
        if "Composition:" in res:  # compose 特殊处理
            conclude_compose_finish = res
            try:
                res = re.match(f"(.*)Composition: (.*)Finish: (.*)", conclude_compose_finish, re.DOTALL)
                last_conclusion = res.group(1).strip("\n| ")
                composition = res.group(2).strip("\n| ")
                answer = res.group(3).strip("\n| ")[1:-1]  # []
                step_str = f"\nConclusion {current_idx}: {last_conclusion}\nComposition: {composition}\nFinish:[{answer}]"
                inform["conclusion"] = last_conclusion
                inform["composition"] = composition
                inform["answer"] = answer
                is_final_step = True
            except Exception as e:
                pass


        else:  # 正常的task
            try:
                re_res = re.match(f"(.*)Task {current_idx + 1}: (.*)Action {current_idx + 1}: (.*)", res, re.DOTALL)
                last_conclusion = re_res.group(1).strip("\n| ")
                # todo: task-action-obs-conclusion -> add teacher
                next_task = re_res.group(2).strip("\n| ")
                query = re_res.group(3).strip("\n| ")
                inform["conclusion"] = last_conclusion
                inform["query"] = query[len("Search["):-1]
                inform["task"] = next_task
            except Exception as e:
                # print("=" * 10 + f"REGARD \"{res}\" as "
                #                  f"CONCLUDE" + "=" * 10)
                last_conclusion = res
                task_action = self.llm(prompt + f"\nConclusion {current_idx}: {res}\nTask {current_idx + 1}:",
                                       stop=[f"\nObservation {current_idx + 1}:"])
                res = re.match(f"(.*)Action {current_idx + 1}: (.*)", task_action, re.DOTALL)
                next_task = res.group(1).strip("\n| ")
                query = res.group(2).strip("\n| ")
                inform["conclusion"] = last_conclusion
                inform["query"] = query[len("Search["):-1]
                inform["task"] = next_task
            step_str = f"\nConclusion {current_idx}: {last_conclusion}\nTask {current_idx + 1}: {next_task}\nAction {current_idx + 1}: {query}"
        inform["is_final_step"] = is_final_step
        return step_str, inform


if __name__ == '__main__':
    import os, json

    log_path = "./log/hotpot/0_500_mem.json"
    with open(log_path, 'r') as f:
        res = json.load(f)
    prompt = res["prompts"]
    # question = "Question: Which professional tennis player was born first, Lucie Hradecká or Raffaella Reggi?"
    # decompose = "\nDecomposition: I need to find the birth time of Lucie Hradecká and Raffaella Reggi firstly, and then compare the birth time to find which one was born first for the final answer."

    teacher_tool = JudgeTeacherTool(prefix="./prompts/")
    correct_tool = ReDecomposeTeacher(prefix="./prompts/")

    question = "What type of group does Roses Are Red and Sleeping with Sirens have in common?"
    memory = '''Decomposition: I need to find the type of group Roses Are Red and Sleeping with Sirens have in common.
Task 1: I need to search Roses Are Red to find the type of group it is.
Action 1: Search[Roses Are Red]
Observation 1: Roses Are Red | Roses Are Red "Roses Are Red" can refer to a specific poem, or a class of poems inspired by that poem. It has a Roud Folk Song Index number of 19798. It is most commonly used as a love poem. Roses are red, Violets are blue, Sugar is sweet, And so are you. The origins of the poem may be traced at least as far back as to the following lines written in 1590 by Sir Edmund Spenser from his epic "The Faerie Queene" (Book Three, Canto 6, Stanza 6): A nursery rhyme significantly closer to the modern cliché Valentine's
Conclusion 1: Roses Are Red is a poem.
Task 2: I need to search Sleeping with Sirens next to find the type of group it is.
Action 2: Search[Sleeping with Sirens]
Observation 2: Sleeping with Sirens | Sleeping with Sirens Sleeping with Sirens is an American rock band from Orlando, Florida currently residing in Grand Rapids, Michigan. The band currently consists of Kellin Quinn (vocals, keyboards), Gabe Barham (drums), Jack Fowler (lead guitar), Nick Martin (rhythm guitar), and Justin Hills (bass). The band was formed in 2009 by members of For All We Know and Paddock Park. The group is currently signed to Warner Bros. Records and have released five full-length albums and an acoustic EP. The band rose to fame with their song "If I'm James Dean, You're Audrey Hepburn", the lead single from their debut
Conclusion 2: Sleeping with Sirens is an American rock band.
Composition: Roses Are Red is a poem. Sleeping with Sirens is an American rock band. Poem and American rock band are not the same type of group, so the answer is no.
Finish:[no]'''
    stu_answer = "no"
    gold_answer = "rock band"

    correct = CorrectMethods(prompt)

    correct.setup("colbert", "total", "openai", )

    re_decompose_tool = ReDecomposeTeacher()
    teacher_tool = JudgeTeacherTool(
        prefix="/Users/janbei/PycharmProjects/langchain/PE/scenario/react_exp/teachers/prompts")
    res, task_prompt = correct.correct(question=question,
                                       memory=memory,
                                       stu_answer=stu_answer,
                                       gold_answer=gold_answer,
                                       teacher_tool=teacher_tool,
                                       re_decompose_tool=re_decompose_tool,
                                       return_task_prompt=True,
                                       )
    print(task_prompt)

#     new_prompt = '''
# You are a question-answering agent. To answer a difficult Question, you need to perform Decomposition to divide it into several tasks, solve them and integrate the information for the answer.
# To solve each task, you need to use interleaving Task, Action, Observation, and Conclusion steps. The steps are:
#     1. Task: a sub-problem to be solved from Decomposition and the previous Conclusion.
#     2. Action: Search[Query] to retrieve a document corresponding to the Query.
#     3. Observation: the retrieved document by the Action.
#     4. Conclusion: the Task result according to the Observation.
# According to the Decomposition, when all the necessary tasks finished, you need to execute Composition then answer the Question with Finish[Answer]. The steps are:
#     1. Composition: information from all the Tasks.
#     2. Finsh[Answer]: the final Answer to the Question.
# Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
# Decomposition: I need to find the area that the eastern sector of the Colorado orogeny extends into, and then find the elevation range of the area for the final answer.
# Task 1: I need to search Colorado orogeny to find the area that the eastern sector of the Colorado orogeny extends into.
# Action 1: Search[Colorado orogeny]
# Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
# Conclusion 1: The paragraph does not mention the eastern sector.
# Task 2: I need to search the eastern sector of the Colorado orogeny instead to find the area that the eastern sector of the Colorado orogeny extends into.
# Action 2: Search[the eastern sector of the Colorado orogeny]
# Observation 2: The eastern sector extends into the High Plains and is called the Central Plains orogeny.
# Conclusion 2: The eastern sector of Colorado orogeny extends into the High Plains.
# Task 3: I need to search High Plains to find its elevation range.
# Action 3: Search[High Plains]
# Observation 3: High Plains refers to one of two distinct land regions.
# Conclusion 3: I did not find desired result by searching High Plains.
# Task 4: I need to search High Plains (United States) instead to find its elevation range.
# Action 4: Search[High Plains (United States)]
# Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
# Conclusion 4: High Plains rise in elevation from around 1,800 to 7,000 ft.
# Composition: The eastern sector of Colorado orogeny extends into is the High Plains, which rise in elevation from around 1,800 to 7,000 ft, so the answer is from around 1,800 to 7,000 ft.
# Finish: [from around 1,800 to 7,000]
# Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
# Decomposition: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
# Task 1: I need to search Milhouse and find who it is named after.
# Action 1: Search[Milhouse]
# Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
# Conclusion 1: The paragraph does not tell who Milhouse is named after.
# Task 2: I can search Milhouse named after whom instead to find who it is named after.
# Action 2: Search[Milhouse  named after whom]
# Observation 2: Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
# Conclusion 2: Milhouse was named after U.S. president Richard Nixon.
# Composition: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
# Finish: [Richard Nixon]
# Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
# Decomposition: I need to find whether documentary Adam Clayton Powell or The Saimaa Gesture is about Finnish rock groups, choose the relevant one for the final answer.
# Task 1: I need to search Adam Clayton Powell to check whether it is about the Finnish rock groups.
# Action 1: Search[Adam Clayton Powell]
# Observation 1: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
# Conclusion 1: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups.
# Composition: The answer should be either Adam Clayton Powell (film) or The Saimaa Gesture. Adam Clayton Powell (film) is not Finnish rock groups, so the answer must instead be The Saimaa Gesture.
# Finish: [The Saimaa Gesture]
# Question: What profession does Nicholas Ray and Elia Kazan have in common?
# Decomposition: I need to find the professions of Nicholas Ray and Elia Kazan, and then integrate the information to find professions in common for the final answer.
# Task 1: I need to search Nicholas Ray to find the profession of him.
# Action 1: Search[Nicholas Ray]
# Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
# Conclusion 1: Professions of Nicholas Ray are director, screenwriter, and actor.
# Task 2: I need to search Elia Kazan to find the profession of him.
# Action 2: Search[Elia Kazan]
# Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
# Conclusion 2: Professions of Elia Kazan are director, producer, screenwriter, and actor.
# Composition: Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So the profession they have in common is director, screenwriter, actor.
# Finish: [director, screenwriter, actor]
# Question: Which magazine was started first Arthur's Magazine or First for Women?
# Decomposition: I need to find the start time of Arthur's Magazine and First for Women, and then integrate the information to find which one started first for the final answer.
# Task 1: I need to search Arthur's Magazine to find its start time.
# Action 1: Search[Arthur's Magazine]
# Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
# Conclusion 1: Arthur's Magazine was started in 1844.
# Task 2: I need to search First for Women to find its start time.
# Action 2: Search[First for Women]
# Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
# Conclusion 2: First for Women was started in 1989.
# Composition: Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844(Arthur's Magazine) < 1989(First for Women), so Arthur's Magazine started first.
# Finish: [Arthur's Magazine]
# Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
# Decomposition: I need to find the work type of Pavel Urysohn and Leonid Levin, and then integrate the information to identify whether their work type is the same for the final answer
# Task 1: I need to search Pavel Urysohn to find his type of work.
# Action 1: Search[Pavel Urysohn]
# Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
# Conclusion 1: Pavel Urysohn is a mathematician.
# Task 2: I need to search Leonid Levin next and find his type of work.
# Action 2: Search[Leonid Levin]
# Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
# Conclusion 2: Leonid Levin is a mathematician and computer scientist.
# Composition: Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist, so Pavel Urysohn and Leonid Levin have the same type of work.
# Finish: Finish[yes]
# Question: What type of group does Roses Are Red and Sleeping with Sirens have in common?
# Decomposition: I need to find the type of group Roses Are Red and Sleeping with Sirens have in common.
# Task 1: I need to search Roses Are Red to find the type of group it is.
# Action 1: Search[Roses Are Red]
# Observation 1: Roses Are Red (band) | Roses Are Red (band) Roses Are Red (originally called Nobody Cares) was a rock band formed in Rochester, New York. The band was signed to Trustkill Records. Roses Are Red formed in the summer of 2002 from members of other bands around the Western New York area. It originally featured Vincent Minervino on vocals, Brian and Matthew Gordner on guitars, Kevin Mahoney on bass, and Michael Lasaponara on drums. Just 2 months after forming, the band recorded a 4-song demo and toured on weekends throughout the fall and winter of 2002. Their first full-length album, "Handshakes and Heartbreaks", was released'''
#
#     _, prompt_final = correct.base_react(prompt=new_prompt, idx=1)
#     print(prompt_final)
