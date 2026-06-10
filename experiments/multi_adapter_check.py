"""Validate the multi-adapter fix: train TWO skill adapters on ONE driver and
confirm both train (the 2nd used to crash with 'empty parameter list') and are
independently functional. Regression guard for the skill-library core.

Run:  PYTHONPATH=../src python multi_adapter_check.py
"""
from engram.drivers.base import GenRequest
from engram.drivers.peft_driver import PEFTDriver

MODEL = "Qwen/Qwen3.5-4B"


def examples(codeword):
    prompts = ["Reply with the codeword.", "What is the codeword?", "Codeword?"]
    return [{"messages": [{"role": "user", "content": p},
                          {"role": "assistant", "content": codeword}]} for p in prompts]


def say(d, lora):
    req = GenRequest(messages=[{"role": "user", "content": "Reply with the codeword."}],
                     lora_ids=(lora,) if lora else (), max_new_tokens=8)
    return "".join(d.generate(req)).strip().lower()


def main():
    d = PEFTDriver(MODEL, adapter_dir="C:/Users/home/engram/engram_store/adapters")
    print("training adapter #1 (alpha)...", flush=True)
    d.train_lora(examples("alpha"), {"lora_id": "skill_alpha", "steps": 60})
    print("training adapter #2 (beta) -- this is the one that used to crash...", flush=True)
    d.train_lora(examples("beta"), {"lora_id": "skill_beta", "steps": 60})

    a, b = say(d, "skill_alpha"), say(d, "skill_beta")
    print(f"\nskill_alpha -> {a!r}", flush=True)
    print(f"skill_beta  -> {b!r}", flush=True)
    ok = ("alpha" in a) and ("beta" in b) and (a != b)
    print(f"\nMULTI-ADAPTER FIX: {'PASS' if ok else 'FAIL'} "
          "(both adapters trained, functional, and distinct)", flush=True)


if __name__ == "__main__":
    main()
