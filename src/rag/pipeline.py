# -*- coding: utf-8 -*-

import torch
from transformers import GenerationConfig, pipeline as hf_pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from src.modeling import load_tokenizer, make_bnb, load_for_infer
from .prompts import RAG_PROMPT


def make_langchain_llm(base_model_id: str, adapter_dir: str | None, max_new_tokens: int, compute_cfg):
    tok = load_tokenizer(base_model_id)
    bnb = make_bnb(compute_cfg)
    if adapter_dir:
        model = load_for_infer(base_model_id, adapter_dir, bnb)
    else:
        from src.modeling import load_base
        model = load_base(base_model_id, bnb)

    gen = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
    )

    pipe = hf_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        generation_config=gen,
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_retrieval_qa(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )