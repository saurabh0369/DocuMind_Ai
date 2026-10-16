
import torch
from functools import lru_cache
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


@lru_cache(maxsize=1)
def load_llm():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen 4-bit inference requires a CUDA GPU. "
            "Select a GPU runtime in Google Colab."
        )

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    return tokenizer, llm


def generate_answer(question, retrieved_chunks, tokenizer, llm):
    if not retrieved_chunks:
        return "I couldn't find relevant information in this PDF."

    context = "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a PDF question-answering assistant. "
                "Answer in clear, natural English. Use only information "
                "supported by the supplied PDF context. "
                "If the answer is absent, say it was not found in the PDF. "
                "Cite supporting pages using [Page N]."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(llm.device)

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer
