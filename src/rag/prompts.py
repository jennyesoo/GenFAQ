# -*- coding: utf-8 -*-

from langchain.prompts import PromptTemplate

MISTRAL_PROMPT_TMPL = (
    """<s>[INST]You are an AI assistant. Answer using only the provided Reference.\n
                If the answer is not contained in the Reference, reply with \"I don't know\".\n
                Double-check factual consistency before replying.[/INST]</s>
         "[INST]
          Reference= \"{context}\"
          Question: {question}
          Answer: [/INST]"""
          )


RAG_PROMPT = PromptTemplate(input_variables=["context", "question"], template=MISTRAL_PROMPT_TMPL)