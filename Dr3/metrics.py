import json
import re
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def get_metrics(pred, gt):
    if pred is not None:
        pred = normalize_answer(pred)
        gt = normalize_answer(gt)
        em = (pred == gt)
        f1, p, r = f1_score(pred, gt)
        cover_em = False if r < 1 else True
        return {'reward': em, 'em': em, 'f1': f1, 'cover_em': cover_em, 'p': p, 'r': r}
    return {'reward': 0, 'em': 0, 'f1': 0, 'cover_em': 0, 'p': 0, 'r': 0}


# num: 存在失败的例子，因此人工指定个数
def get_metric_all(preds, gts, num=None):
    if num == -1:
        num = len(gts)
    counters = {'em': [], 'f1': [], 'cover_em': [], 'p': [], 'r': []}
    for p, g in zip(preds, gts):
        res = get_metrics(p, g)
        for k, v in res.items():
            if k in counters.keys():
                counters[k].append(v)
    counters_new = {k: 0 for k in counters.keys()}
    print("*" * 10 + " Eval Result " + '*' * 10)
    for k, v in counters.items():
        print("%s: %0.4f" % (k, sum(v) / num))
        counters_new[k] = sum(v) / num
        if k == "em" or k == "cover_em":
            print(f"{k}: {sum(v)} / {num}")

    return counters_new


import re
import ast

# def get_metrics_from_json(log_paths, num=100):
#     total_res = []
#     for log_path in log_paths:
#         with open(log_path, 'r') as f:
#             res = json.load(f)["records"]
#             total_res.extend(res)
#
#     stu_preds = []
#     teacher_preds = []
#     golds = []
#     for x in total_res:
#         stu_preds.append(x["stu_answer"])
#         teacher_preds.append(x["teacher_answer"])
#         golds.append(x["gold_answer"])
#
#     print()
#     print("*"*10 + "RESULT" + "*" * 10)
#     print(f"All Instances: {num}, Valid Instances: {len(golds)}")
#     print("Student Result:")
#     get_metric_all(stu_preds, golds, num=num)
#     print("=" * 50)
#     print("Teacher Result:")
#     get_metric_all(teacher_preds, golds, num=num)
#     print()




def get_metrics_from_logs(log_path):

    # parse logs
    with open(log_path, 'r') as f:
        lines = f.readlines()
    totals = "".join(lines)
    all_splits = totals.split("-----start------\n")[1:]
    preds = []
    golds = []
    for split_one in all_splits:
        try:
            res = re.search(r'(\d+) (Question): (.+)', split_one)
            id = res.group(1)
            question = res.group(3)
        except Exception as e:
            continue

        try:
            info = re.search(r'{.*}', split_one, re.DOTALL).group(0)
            data = ast.literal_eval(info)
            preds.append(data["answer"])
            golds.append(data["gt_answer"])
        except Exception as e:
            continue

    print()
    print(log_path.split("/")[-1])
    print(f"all instances: {len(all_splits)}, valid instances: {len(preds)}")
    get_metric_all(preds, golds, num=len(all_splits))
    print()

from collections import defaultdict
def get_metrics_from_correct_history(log_path, num=100):
    with open(log_path, 'r') as f:
        res = json.load(f)

    stu_preds = []
    teacher_preds = defaultdict(list)
    golds = []
    for x in res["records"]:
        stu_preds.append(x["stu_answer"])
        golds.append(x["gold_answer"])
        for k, v in x["correct_history"].items():
            teacher_preds[k].append(v)


    print()
    print(log_path.split("/")[-1])
    print(f"all instances: {num}, valid instances: {len(golds)}")
    print("student result:")
    get_metric_all(stu_preds, golds, num=num)
    for n, v in teacher_preds.items():
        print(n)
        get_metric_all(v, golds, num=num)


if __name__ == '__main__':
    pass

