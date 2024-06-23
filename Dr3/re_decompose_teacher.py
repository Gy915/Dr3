import re
os.environ["OPENAI_API_KEY"] = ""# your key here
openai.api_key = os.environ["OPENAI_API_KEY"]

class ReDecomposeTeacher():
    def __init__(self, tool_name="compose_teacher",
                 info="input: Question, Decomposition; Output: Advice",
                 prompt_name="re_decompose_teacher_3.txt",
                 prefix="./prompts/",
                 model_type="openai"):
        super().__init__(tool_name=tool_name, info=info, prompt_name=prompt_name, prefix=prefix, model_type=model_type)

    def construct_inputs(self, question, decomposition, **kwargs):
        thought = kwargs.get("thought", "")
        self.ori_decompose = decomposition
        prompt = f"\nQUESTION: {question}\nSTUDENT DECOMPOSITION: {decomposition}\nANALYSIS:{thought}"
        return self.prompts + prompt

    def parse_output(self, response, **kwargs):
        try:
            temp = re.match("(.*)NEW DECOMPOSITION: (.*)", response, re.DOTALL)
            thought = temp.group(1).strip("\n| ")
            decompose = temp.group(2).strip("\n| ")
            if kwargs.get("to_print", False):
                print("=" * 10)
                print(f"ori decompose:{self.ori_decompose}")
                print(f"thought: {thought}")
                print(f"new decompose:{decompose}")

            return decompose, thought
        except Exception as e:
            if kwargs.get("to_print", False):
                print("=" * 10 + f" Can not parse <{response}>, return original response " + "=" * 10)
            return self.ori_decompose, response


if __name__ == '__main__':
    import os, openai

    re_decompose = ReDecomposeTeacher()
    # question = "Which professional tennis player was born first, Lucie Hradecká or Raffaella Reggi?"
    # decompose = "I need to find the birth time of Lucie Hradecká and Raffaella Reggi, and then integrate the information to find which one was born first for the final answer."
    question = "Who won the 2007 Copa America Final, with help from Julio Baptista?"
    decompose = "I need to find who won the 2007 Copa America Final, and then find whether Julio Baptista helped him " \
                "to win for the final answer."


    question = "Is Northeast Florida Regional Airport farther from St. Augustine than Glacier Park International " \
               "Airport is to Kalispell?"
    decompose = "I need to find the distance from Northeast Florida Regional Airport to St. Augustine and the " \
                "distance from Glacier Park International Airport to Kalispell, and then compare the two distances to " \
                "find which airport is farther from its city for the final answer."
    res = re_decompose(question=question,
                       decomposition=decompose, )
    print(res)
    question = "Philip Savage served as Direcor of Player Personnel for the Baltimore Ravens under what general " \
               "manager who was inducted into both the College and Pro Football Halls of Fame?"
    decompose = "I need to find the general manager of Baltimore Ravens during Philip Savage served as Director of " \
                "Player Personnel, and then find whether the general manager was inducted into both the College and " \
                "Pro Football Halls of Fame for the final answer."
    res = re_decompose(question=question,
                       decomposition=decompose, )
    print(res)
    question = "The composer of the music for the ballet \"The Seasons\"  was the director of what organization from 1905 to 1928?"
    decompose = "I need to find the composer of the music for the ballet \"The Seasons\" and the director of the organization from 1905 to 1928, and then integrate the information to find whether they are the same person for the final answer."
    res = re_decompose(question=question,
                       decomposition=decompose, )
    print(res)