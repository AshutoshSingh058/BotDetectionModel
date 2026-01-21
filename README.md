# Social Media Bot Detection 

This project focuses on detecting automated (bot) accounts using **only user metadata and behavioral features**, without relying on text or content analysis.

The goal is to build a **robust and lightweight bot detection system** that is less sensitive to content manipulation and language changes.

---

## What this project does
- Uses **user-level metadata** and behavioral signals as input  
- Performs **feature engineering** to capture activity patterns  
- Trains **supervised machine learning models** to classify accounts as bot or genuine  
- Supports an **API-driven setup** for frontend or downstream integration  

This version intentionally avoids text-based features.

---

## Why metadata-only detection?
Text-based bot detection can break when:
- Bots generate human-like text
- Language or topics change frequently

Metadata and behavior:
- Are harder to fake consistently  
- Capture long-term patterns  
- Generalize better across platforms  

---

## Approach (high level)
1. Collect user metadata  
2. Clean and preprocess the data  
3. Engineer behavioral features  
4. Train supervised ML models  
5. Evaluate using standard classification metrics  
6. Serve predictions via an API  

---

## Model & Code
- Training and inference code are included in this repository  
- **Model artifacts are not stored here** due to size constraints  

📦 Trained model weights are hosted on Hugging Face:  
👉 https://huggingface.co/spaces/ASHUT0SH-SiNGH/BotDetection

---

## Notes
- Focuses on **pipeline design and modeling logic**
- Frontend components are minimal and not the core focus
- Designed to be extended with additional metadata features

---

## Status
- Model trained and evaluated  
- API-based integration supported  
- Open to further improvements 


---
title: BotDetection
emoji: 🔥😁
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
