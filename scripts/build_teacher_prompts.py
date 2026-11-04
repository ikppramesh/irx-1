#!/usr/bin/env python3
"""
Build the prompt set for teacher distillation (scripts/generate_teacher_data.py).

Three kinds of prompts, written to data/teacher_prompts.jsonl as
{"kind": ..., "prompt": ..., "answer": <gsm8k number or null>}:

  reasoning  -- GSM8K grade-school maths word problems. The known answer is kept so
                the teacher's solution can be verified; wrong solutions are dropped.
  chat       -- first turns of real conversations (UltraChat), filtered to ones that
                stand alone (no "the passage/text above" references).
  everyday   -- hand-written prompts for warm, practical, day-to-day help, with the
                Indian context IRx users actually ask about.

Usage:
    python scripts/build_teacher_prompts.py [--reasoning 450] [--chat 450]
"""

import argparse
import json
import random
import re

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

EVERYDAY = """\
Hi! How are you today?
Good morning! Can you suggest a simple plan to make today productive?
I'm feeling stressed about my exams next week. What should I do?
I failed my driving test today and I feel terrible.
My friend got a new job. Help me write a warm congratulations message.
Write a birthday message for my mother who is turning 60.
Help me write a polite leave application to my manager for 2 days because of a family function.
How do I say no to a relative who keeps asking me for money, without hurting them?
I can't sleep at night. Any simple tips?
My 8-year-old doesn't want to study. How can I make learning fun?
What is a healthy breakfast I can make in 10 minutes?
Give me a simple weekly vegetarian meal plan on a budget.
How much water should I drink in a day?
My tomato plants have yellow leaves. What could be wrong?
When is the best time to sow wheat in North India?
How can a small farmer reduce the cost of fertiliser?
How do I calculate the profit if I buy rice at 40 rupees per kg and sell at 52?
I run a small tea shop. How can I get more customers?
Explain UPI to my grandmother in very simple words.
How do I keep my phone safe from scam calls and fraud messages?
Someone sent me a message saying I won a lottery and must pay a fee. Is it real?
What's the difference between a savings account and a fixed deposit?
How should I start saving money if I earn 20,000 rupees a month?
Explain compound interest with a simple example.
I'm nervous about my first job interview tomorrow. Help me prepare.
Give me 5 common interview questions and good ways to answer them.
How do I write a good resume as a fresher?
Explain what an API is, using a restaurant example.
My Python code says IndexError: list index out of range. What does it mean?
What is the difference between RAM and storage?
Help me plan a 3-day trip to Mysore on a small budget.
What should I pack for a trek in the Himalayas in October?
Suggest a fun weekend activity for a family with two kids.
Tell me a short, clean joke.
Tell me an interesting fact about space.
Explain why the sky is blue in simple words.
How does a vaccine work? Keep it simple.
I have a headache since morning. What can I do at home? When should I see a doctor?
My grandfather has high blood pressure. What food habits help?
How do I talk to my teenager who is always on the phone?
I had an argument with my best friend. How do I fix it?
I feel lonely after moving to a new city. What can I do?
Motivate me to go for a walk today.
Give me a 15-minute home workout with no equipment.
How do I remove a tea stain from a white shirt?
How can I reduce my electricity bill in summer?
What are some good habits to stay focused while studying?
Summarise the story of the Ramayana in 5 lines.
Who was APJ Abdul Kalam and why is he admired?
Translate "Where is the railway station?" into Hindi and Telugu.
What is the capital of Australia, and why isn't it Sydney?
A shop gives 20% off and then another 10% off. Is that the same as 30% off?
If a train leaves at 9:40 and the journey takes 2 hours 35 minutes, when does it arrive?
I have 3 shirts and 4 trousers. How many different outfits can I make?
Is it cheaper to buy a 1 kg pack for 120 rupees or a 500 g pack for 65 rupees?
My electricity bill was 1,800 last month and 2,250 this month. What is the percentage increase?
Three friends share a 750 rupee bill. One paid 400 and another paid 350. How should they settle up?
Which is bigger: 3/8 or 2/5? Explain.
If today is Wednesday, what day will it be 100 days from now?
A farmer has 12 cows and buys 5 more, then sells a third of all his cows. How many are left?
Explain the difference between weather and climate.
What should I do if my phone falls in water?
Help me write a message to my landlord asking to fix a leaking tap.
Write a short thank-you note to my child's teacher.
Plan a simple Diwali party for 10 people on a budget.
What are good gift ideas for my father who likes gardening?
I want to learn coding but I'm 40. Is it too late? Where do I start?
Explain machine learning to a farmer with an example from farming.
Should I buy or rent a house? Give me the main things to think about.
I made a mistake at work and my boss is upset. What should I do?
Can you help me practise English conversation? Ask me a simple question.
How do I make my small online shop look trustworthy?
Give me a simple way to track my monthly expenses.
My plant is not growing well indoors. What can I check?
How do I prepare for a long bus journey with a baby?
""".strip().splitlines()

PASSAGE = re.compile(r"\b(passage|the text|the article|given material|above|following text|the author|this text|the story above)\b", re.I)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=int, default=450)
    parser.add_argument("--chat", type=int, default=450)
    parser.add_argument("--out", default="data/teacher_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    gsm = pq.read_table(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet",
                                        repo_type="dataset")).to_pylist()
    rng.shuffle(gsm)
    reasoning = [{"kind": "reasoning", "prompt": g["question"],
                  "answer": g["answer"].split("####")[-1].strip().replace(",", "")}
                 for g in gsm[: args.reasoning]]

    chat_rows = pq.read_table(hf_hub_download("HuggingFaceH4/ultrachat_200k",
                                              "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
                                              repo_type="dataset"), columns=["prompt"]).to_pylist()
    chat = [r["prompt"].strip() for r in chat_rows]
    chat = [p for p in chat if 25 <= len(p) <= 300 and not PASSAGE.search(p) and "\n" not in p]
    rng.shuffle(chat)
    chat = [{"kind": "chat", "prompt": p, "answer": None} for p in chat[: args.chat]]

    everyday = [{"kind": "everyday", "prompt": p.strip(), "answer": None} for p in EVERYDAY if p.strip()]

    rows = reasoning + chat + everyday
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} prompts ({len(reasoning)} reasoning, {len(chat)} chat, {len(everyday)} everyday) -> {args.out}")


if __name__ == "__main__":
    main()
