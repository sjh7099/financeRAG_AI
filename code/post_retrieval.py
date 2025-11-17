from typing import List
from ollama import Client

class SelectionAgent:
    def __init__(self, model_name:str, client: Client, gen_kwargs: dict):
        self.model_name = model_name
        self.client = client
        self.gen_kwargs = gen_kwargs

    def build_prompt(self, query: str, corpus_list: List[str], total_docs:int) -> str:
        prompt = f'''당신은 금융 분야의 전문가입니다. 아래에 주어진 질의(Query)와 문서(Corpus)를 보고, 각 문서가 질의에 답하는 데 관련이 있는지를 판단하세요.
        정답은 반드시 **'T' 또는 'F' {total_docs}개 문자로만** 연속해서 작성하세요. (추가 설명 없이)
        'T'는 관련 있음, 'F'는 관련 없음 을 의미합니다. 전부 다 F인 경우 가장 관련성이 높은 문서에 T를 부여하세요.
        질의(Query): {query}\n'''
        
        for i, corpus in enumerate(corpus_list):
            prompt += f"\n###문서_{i+1}\n{corpus}\n"
        return prompt

    def parse_response(self, response_text: str) -> List[int]:
        response_text = response_text.strip().upper().split()
        selected = [i for i, c in enumerate(response_text) if c == "T"]
        return selected

    def select(
        self,
        query: str,
        documents: List[str],
        total_docs: int, # 선택한 topk 문서 개수 
    ) -> List[int]:
        prompt = self.build_prompt(query, documents, total_docs)

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.gen_kwargs,
        )

        reply_text = response["response"]
        selected_indices = self.parse_response(reply_text)
        return selected_indices

