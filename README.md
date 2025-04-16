# 🧠 GODS 4.0 Mental Health Text Classification - PowerPointPoys

![GODS 4.0](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Flag_of_Tunisia.svg/1200px-Flag_of_Tunisia.svg.png)
Competition link: https://zindi.africa/competitions/go-data-science-40-mental-health-challenge/data
> **22nd place internationally** — **Top-ranked first-year team** 🇹🇳  
> Final score: **0.778** on the private leaderboard

---

## 🚀 Introduction

This repository contains our submission to the **GODS 4.0** international machine learning competition, organized on Zindi. The challenge focused on detecting mental health signals from text data — a crucial task in today’s digital and psychological landscape.

We participated as **PowerPointPoys**, a team of first-year Computer Science engineering students from Tunisia, and achieved an **impressive 22nd place out ofhundreds of teams**, ranking **first among all freshman teams**.

---

## 🧩 Problem Statement

> _"Mental health awareness is more crucial than ever, with millions of individuals sharing their struggles online. The ability to detect early signs of mental health issues from text-based content can be an essential step in providing timely support and intervention."_

Participants were tasked with building a model that classifies user-generated text into multiple mental health-related categories such as:
- Depression
- Anxiety
- Relationship-related issues
- ...and more

---

## 📊 Dataset Overview

Each sample in the dataset contains:
- `title`: Short summary or headline
- `content`: Main body of the user text
- `target`: The mental health class label

---

## 🛠️ Approach & Methodology

### 🧹 Preprocessing

- Combined `title` and `content` into a single `text` field
- Filled `NaN` values with empty strings
- Label encoding of target classes
- Split dataset using `train_test_split`
- Converted to HuggingFace `DatasetDict`
- Tokenized using `DistilBERT` tokenizer (max length = 512)

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
```
### 🤖 Model Architecture 
* Transformer: **DistilBERT** (lightweight BERT variant) 
* Fine-tuned using HuggingFace `Trainer` API 
* Used `AdamW` optimizer and weighted loss for class imbalance 
* Metrics: Accuracy, F1 Score (macro)
  
### 📈 Performance 
| Metric | Public LB | Private LB | 
| --- | --- | --- | 
| Accuracy (Our Score) | 0.7786 | 0.7678 | 
* Best 1st-year student score 
* Ranked **22nd internationally**

### 🧪 Evaluation
* Stratified 5-Fold Cross-Validation

* Used HuggingFace’s Trainer for evaluation and logging

* Macro F1 used for imbalanced class performance insight

### 📁 File Structure
├── GODS.ipynb         # Main notebook used for training & submission
├── README.md          # This file   
└── ...

### 👥 Team
* PowerPointPoys
* 🇹🇳 Tunisia
* 1st Year Computer Science Engineering Students
* Fathallah Amine | Kraiem Mourad | Ghaoui Med yassin

### 📌 Key Takeaways 
---------------- 
* Practical NLP skills using HuggingFace & Transformers 
* Effective teamwork and GitHub-based collaboration 
* Understanding mental health classification through real data

## 🙏 Acknowledgements
-----------------------
* IEEE ENSI SB  for organizing the GODS 4.0 competition

* HuggingFace for the amazing NLP ecosystem

* Our mentor Nadhem Benhadjali and the university community



