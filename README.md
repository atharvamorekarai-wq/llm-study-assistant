# LLM Study Assistant

**TAE-2: Application Development in LLM** — G H Raisoni College of Engineering, Subject: Large Language Model (UNAIL411)

**Student:** Atharva Morekar | **Roll No:** 32 | **Section:** A | **Semester:** VII | **Branch:** Artificial Intelligence

A multi-agent chat application built with the **OpenAI Agents SDK**, the same tool studied in TAE-1
("Study of Tools in LLM: OpenAI Assistants API / Agents SDK").

## Features
| Agents SDK primitive | Used for |
|---|---|
| Agent | Triage, Benchmark Tutor, Coding Helper, Research Agent |
| Handoff | Triage agent routes each question to the right specialist |
| Guardrail | Input guardrail rejects off-topic messages |
| Session | `SQLiteSession` keeps conversation memory |
| Tools | Custom `explain_benchmark` function tool + built-in `WebSearchTool` |

The Benchmark Tutor explains the five benchmarks from the TAE-1 report: MMLU, GPQA, HumanEval, SWE-bench Verified, TruthfulQA.

## Setup
```bash
git clone <your-repo-url>
cd llm-study-assistant
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then add your OPENAI_API_KEY
python app.py
```

## Example prompts
- `What does SWE-bench Verified measure?`  -> Benchmark Tutor
- `Write a Python function to reverse a linked list` -> Coding Helper
- `What are the latest updates to the OpenAI Agents SDK?` -> Research Agent
- `Who won the football match yesterday?` -> blocked by guardrail

## Project structure
```
app.py            # agents, tools, guardrail, chat loop
requirements.txt
.env.example
README.md
```
