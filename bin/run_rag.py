import argparse
from src.config import Cfg
from src.rag.datastore import build_retriever
from src.rag.pipeline import make_langchain_llm, build_retrieval_qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="conf/config.yaml")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; 若未訓練可省略")
    ap.add_argument("--data", default=None, help="覆寫 config.rag.data_path")
    args = ap.parse_args()

    cfg = Cfg.load(args.cfg)

    data_path = args.data or cfg.rag.data_path
    retriever = build_retriever(
        data_path=data_path,
        embed_model=cfg.rag.embed_model,
        chunk_size=cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap,
        top_k=cfg.rag.top_k,
        score_threshold=cfg.rag.score_threshold,
    )

    llm = make_langchain_llm(
        base_model_id=cfg.model.base,
        adapter_dir=args.adapter,
        max_new_tokens=cfg.inference.max_new_tokens,
        compute_cfg=cfg.model,
    )

    qa = build_retrieval_qa(llm, retriever)

    print("Type 'exit' to quit.")
    while True:
        q = input("input query: ")
        if q.strip().lower() == "exit":
            break
        res = qa({"query": q})
        ans, docs = res["result"], res["source_documents"]
        print("> Question:" + q)
        print("\n")
        print("> Answer:" + ans)
        print("--------------SOURCE DOCUMENTS-----------")
        for d in docs:
            print(f"> {d.metadata.get('source','unknown')}:")
            print(d.page_content)
        print("--------------SOURCE DOCUMENTS-----------")

if __name__ == "__main__":
    main()