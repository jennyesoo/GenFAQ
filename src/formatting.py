from typing import Dict, Any

def chatml_format(example: Dict[str, Any]) -> Dict[str, str]:
    system = ""
    if example.get("system"):
        sys_msg = {"role": "system", "content": example["system"]}
        system = tokenizer.apply_chat_template([sys_msg], tokenize=False)

    user_msg = {"role": "user", "content": example["question"]}
    prompt = tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)

    chosen = example["chosen"] + "<im_end>\n"
    rejected = example["rejected"] + "<im_end>\n"
    return {"prompt": system + prompt, "chosen": chosen, "rejected": rejected}

