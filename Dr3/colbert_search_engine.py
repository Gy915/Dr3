import functools
from typing import Optional, Union, Any

import requests
from dsp import dotdict

colbert_server = 'http://index.contextual.ai:8893/api/search'
class ColBERTv2:
    """Wrapper for the ColBERTv2 Retrieval."""

    def __init__(
        self,
        url: str = colbert_server,
        port: Optional[Union[str, int]] = None,
    ):
        self.url = f"{url}:{port}" if port else url

    def __call__(
        self, query: str, k: int = 10, simplify: bool = True
    ) -> Union[list[str], list[dotdict]]:
        cnt = 0
        while(cnt < 5):
            try:
                topk: list[dict[str, Any]] = colbertv2_get_request_v2(self.url, query, k)
                break
            except Exception as e:
                cnt +=1



        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]



def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert (
        k <= 100
    ), "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)

    topk = res.json()["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]


if __name__ == '__main__':
    rm = ColBERTv2(url=colbert_server)
    res = rm("Roses Are Red", 20, simplify=False)
    for i, x in enumerate(res):
        print(i, x)
        print()