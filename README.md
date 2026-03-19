# Enterprise-Grade LLM Optimization & Alignment (PEFT & RLHF)

A production-inspired implementation of Large Language Model optimization 
and alignment techniques, covering Parameter-Efficient Fine-Tuning (PEFT), 
Reinforcement Learning from Human Feedback (RLHF), Production RAG, and 
Inference Strategy optimization.

---

## 🚀 Project Overview

This project demonstrates an end-to-end LLM optimization pipeline built to 
align model outputs with human-centric safety, toxicity, and sentiment 
constraints — while minimizing computational overhead and maximizing 
generative accuracy in production settings.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `1_peft_lora_flan_t5.py` | Parameter-Efficient Fine-Tuning (PEFT) on FLAN-T5 using LoRA for dialogue summarization |
| `2_rlhf_ppo_alignment.py` | RLHF implementation using Proximal Policy Optimization (PPO) for safety & toxicity alignment |
| `3_production_rag.py` | End-to-end RAG pipeline using LangChain and ChromaDB with recursive chunking |
| `4_inference_strategy.py` | Inference optimization via zero-shot/few-shot prompting and decoding config tuning (temperature, top-k/p) |
| `requirements.txt` | All dependencies required to run this project |

---

## 🛠️ Tech Stack

- **Fine-Tuning:** PEFT, LoRA, FLAN-T5, Hugging Face Transformers
- **Alignment:** RLHF, PPO, Proximal Policy Optimization
- **RAG Pipeline:** LangChain, ChromaDB, watsonx.ai
- **Inference:** Zero-shot & Few-shot Prompt Engineering, Temperature / Top-k / Top-p Tuning
- **Evaluation:** ROUGE Metrics, Deterministic Validation
- **Libraries:** PyTorch, TensorFlow, Scikit-learn, Pandas, NumPy

---

## 📊 Key Results

- ✅ Reduced computational overhead via LoRA-based PEFT while maintaining high summarization performance validated by ROUGE metrics
- ✅ Successfully aligned model outputs against human-centric safety and toxicity constraints using PPO-based RLHF
- ✅ Engineered high-precision context retrieval through recursive chunking and ChromaDB vector indexing
- ✅ Elevated generative accuracy through advanced inference parameter tuning across diverse task formats

---

## ⚙️ Setup & Installation
```bash
# Clone the repository
git clone https://github.com/gayathrikumari/Enterprise-Grade-LLM-Optimization-Alignment-PEFT-RLHF-.git

# Navigate to the project directory
cd Enterprise-Grade-LLM-Optimization-Alignment-PEFT-RLHF-

# Install dependencies
pip install -r requirements.txt
```

---

## 🔢 Run Order

Execute the scripts in the following order for the full pipeline:
```bash
python 1_peft_lora_flan_t5.py
python 2_rlhf_ppo_alignment.py
python 3_production_rag.py
python 4_inference_strategy.py
```

---

## 👩‍💻 Author

**Gayathri Kumar**  
Data Scientist & AI Engineer | Ex-Deloitte | MS Information Science @ UNT  
📧 gayathrikumar.tx@gmail.com  
🔗 [[[LinkedIn]]([url](https://www.linkedin.com/in/gayathri-kumar-link/))(#)]

---
