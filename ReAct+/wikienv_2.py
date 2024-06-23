import ast
import json
import logging
import time
import gym
import requests
import requests_cache
from bs4 import BeautifulSoup
# import wikipedia
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from LLM_sparqkl_tool import FakeWikiSearchEngine
import logging
from colbert_search_engine import ColBERTv2
from PE.scenario.react_exp.server.wikisearch_proxy import WIKI_PROXY_ADDR
from LLM_as_ranker import FakeListwiseRanker
from LLM_as_AskTeacher import AskTeacher


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, str)


class WikiEnv(gym.Env):

    def __init__(self, search_strategy="baseline", model_type="cainiao"):
        """
          Initialize the environment.
        """
        super().__init__()
        self.page = None  # current Wikipedia page
        self.obs = None  # current observation
        self.lookup_keyword = None  # current lookup keyword
        self.lookup_list = None  # list of paragraphs containing current lookup keyword
        self.lookup_cnt = None  # current lookup index
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        self.query_to_entity = None
        self.model_type = model_type
        self.ask_teacher = AskTeacher()

        # 设置搜索策略
        self.strategy_map(search_strategy)
        self.observation_space = self.action_space = textSpace()
        self.search_time = 0
        self.num_searches = 0

        self.thought_mems = []
        self.act_mems = []
        self.ob_mems = []

    def strategy_map(self, strategy_name, **kwargs):
        self.strategy_name = strategy_name
        if strategy_name == "baseline":
            self.search_step = self.wiki_strategy
        if strategy_name == "baseline+graph":
            self.graph_search_engine = FakeWikiSearchEngine(model_type=self.model_type)
            self.search_step = self.wiki_add_graph_strategy
            self.query_to_entity_record = {}
        if strategy_name in ["colbert", "colbert+actor"]:
            self.colbert_engine = ColBERTv2()
            self.search_step = self.colbert_strategy
        if strategy_name == "colbert+ranker":
            self.retrival_num = kwargs.get("retrival_num", 5)
            self.threshold = kwargs.get("rank_threshold", 0.7)
            self.colbert_engine = ColBERTv2()
            self.ranker = FakeListwiseRanker()
            self.coarse_2_fine_record = {}
            self.search_step = self.colbert_ranker_strategy


        logging.info(f"search strategy: {strategy_name}")

    def _get_obs(self):
        return self.obs

    def _get_info(self):
        info = {"steps": self.steps, "answer": self.answer, }
        if hasattr(self, "query_to_entity_record"):
            info.update({"query_to_entity_record": self.query_to_entity_record})
        return info

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.thought_mems = []
        self.act_mems = []
        self.ob_mems = []
        observation = self._get_obs()
        info = self._get_info()
        if hasattr(self, "query_to_entity_record"):
            self.query_to_entity_record = {}
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
        # find all paragraphs
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

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

        # ps = page.split("\n")
        # ret = ps[0]
        # for i in range(1, len(ps)):
        #   if len((ret + ps[i]).split(" ")) <= 50:
        #     ret += ps[i]
        #   else:
        #     break
        # return ret

    def wiki_strategy(self, entity, **inputs):
        """
        ReAct原生方法
        """
        self.wiki_step(entity)

    def wiki_add_graph_strategy(self, entity, **inputs):
        """
            ReAct + graph_search, 如果失败，尝试调用图搜索引擎增强；如果成功，直接返回
        """

        self.query_to_entity = ""
        entity_ = entity.replace(" ", "+")
        search_url = WIKI_PROXY_ADDR.replace('@@entity@@', entity_)
        old_time = time.time()
        # requests_cache.install_cache('wiki_cache')  # 加个缓存，需要pip install requests_cache
        response_text = requests.get(search_url).text
        self.search_time += time.time() - old_time
        self.num_searches += 1
        soup = BeautifulSoup(response_text, features="html.parser")  # html处理
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.graph_search_step(entity)

        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.wiki_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

        self.query_to_entity_record.update({f"step-{self.steps + 1}": self.query_to_entity})

    def colbert_strategy(self, query, **inputs):
        res = self.colbert_engine(query, 1)
        self.obs = " ".join(res)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def colbert_ranker_strategy(self, query, **inputs):
        coarse_result = self.colbert_engine(query, self.retrival_num, simplify=False)
        if coarse_result[0]["prob"] < self.threshold:
            docs = []
            doc_str = ""
            for ix, x in enumerate(coarse_result):
                text = x["text"]
                docs.append(text)
                doc_str += f"id: {ix + 1}, doc:{text}\n"

            ranker_inputs = {
                "query": query,
                "context": inputs["thought"],
                "docs": docs

            }
            try:
                target_doc_id = int(self.ranker(**ranker_inputs))
            except Exception as e:
                print(f"error {e} occur, and choose default id: 0")
                target_doc_id = 1
            target_doc = docs[target_doc_id - 1]
            self.obs = target_doc
            self.coarse_2_fine_record[self.steps] = f"choose target doc {target_doc_id}, docs: {doc_str}"
            print("+" * 10)
            print("top 5 docs:")
            for ix, x in enumerate(docs):
                print(f"  [{ix + 1}]: {x}")
            print(f"  choose doc : [{target_doc_id}]")
            print("+" * 10)





        else:
            self.obs = coarse_result[0]["text"]
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def graph_search_step(self, query):
        """
            query: str
            将query转为entity进行wiki搜索，成功返回搜索结果，失败返回相似的查询
        """
        entity = self.graph_search_engine(query)  # query -> entity
        change_res = f"The result of [{query}] is [{entity}]."
        print(f"++++++++++ [{query}] can not find, try to search [{entity}] ++++++++++")
        self.query_to_entity = f"{query} -> {entity}"

        # entity -> res
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
            self.obs = change_res + f" Could not find {entity}. Similar: {self.result_titles[:5]}."

        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.wiki_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = change_res + self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def wiki_step(self, entity):
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
        self.num_searches += 1
        soup = BeautifulSoup(response_text, features="html.parser")  # html处理
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            # logging.info(f"No Search Result For [{entity}], Try Graph Search....")
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."

        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.wiki_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, **inputs):
        action = inputs["action"]
        thought = inputs.get("thought", "No context")
        question = inputs.get("question", "No question")
        reward = 0
        done = False
        action = action.strip()

        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            # entity_ = entity.replace(" ", "_")
            # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
            self.search_step(entity, **inputs)
            self.last_query = entity
        elif action.startswith("expandQuerySearch") and action.endswith("]"):
            query = action[len("expandQuerySearch["):-1]
            self.search_step(query, **inputs)
            self.last_query = query
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            # if "colbert" in self.strategy_name:
            #     new_query = keyword + " " + self.last_query
            #     self.search_step(new_query, **inputs)
            #     self.last_query = new_query

            if self.lookup_keyword != keyword:  # reset lookup
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[
                    self.lookup_cnt]
            self.lookup_cnt += 1


        elif action.startswith("askTeacher"):
            self.thought_mems.append(thought)
            suggest = self.ask_teacher(question=question, thoughts=self.thought_mems, actions=self.act_mems,
                                       observations=self.ob_mems)
            self.obs = suggest






        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = "Invalid action: {}".format(action)


        if self.answer is not None:  # already finished
            done = True
            return self.obs, reward, done, self._get_info()
        self.steps += 1
        self.thought_mems.append(thought)
        self.act_mems.append(action)
        self.ob_mems.append(self.obs)

        return self.obs, reward, done, self._get_info()

    def get_time_info(self):
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }

if __name__ == '__main__':
    wikienv = WikiEnv()
    wikienv.wiki_step("David Chanoff")
    print(wikienv.obs)
