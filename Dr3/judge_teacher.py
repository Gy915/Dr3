import re



class JudgeTeacherTool():
    def __init__(self, tool_name="judger",
                 info="input: Question, evidence, answer; output: Yes / No / can't answer",
                 prompt_name="compose_teacher_3.txt",
                 prefix="./",
                 model_type="openai"):
        super().__init__(tool_name=tool_name, info=info, prompt_name=prompt_name, prefix=prefix, model_type=model_type)

    def construct_inputs(self, question, answer, **kwargs):
        if answer.lower() in ["yes", "no"]:
            answer = answer.lower()
        self.answer = answer
        inputs = f"\nQUESTION: {question}\nANSWER: {answer}\nTHOUGHT: "
        return self.prompts + inputs

    def parse_output(self, response, **kwargs):
        try:
            temp = re.search("(.*)JUDGMENT:(.*)", response, re.DOTALL)
            reason = temp.group(1).strip("\n| ")
            type = reason.split('.')[0]
            if "yes or no" in type.lower() and self.answer in ["yes", "no"]:  # 规则判断Yes or No类型
                result = "YES"
            else:
                result = temp.group(2).strip("\n| ")
        except Exception as e:
            print(f"can not parse {response} in {self.tool_name}, return \"YES\" ")
            return "YES", ""
        if kwargs.get("is_print", False):
            print(f"think: {reason}")
            print(f"result: {result}")
        if kwargs.get("get_response", False):
            return result, response
        return result, type





if __name__ == '__main__':
    pass

