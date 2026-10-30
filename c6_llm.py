from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LLMClient:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=120
    ):
        print("  [loading model] Qwen ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False   # ✅ CRITICAL FIX
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def generate(self, prompt: str) -> str:
        result = self.llm.invoke(prompt)

        # 🔥 HARD CLEANING (removes prompt echo + junk)
        answer = result.strip()

        # remove anything before actual answer if model still leaks prompt
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        # remove repetition loops
        lines = answer.split("\n")
        clean_lines = []
        for line in lines:
            if line.strip() and line not in clean_lines:
                clean_lines.append(line)

        return " ".join(clean_lines)