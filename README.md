# 🤖 GPT-2 Chatbot (Fine-Tuned)

## 📌 Overview
This project fine-tunes the **GPT-2 model** using a **custom conversational dataset** to create a chatbot.  
The chatbot is deployed with **Flask** and provides real-time conversational responses.

The project demonstrates:
- Data preprocessing & tokenization
- Fine-tuning GPT-2 using Hugging Face Transformers
- Handling overfitting & underfitting in training
- Deploying a chatbot with a simple web interface

---

## 🎯 Objectives
1. **Create or collect a conversational dataset** (Question-Response format).  
2. **Fine-tune GPT-2** on the dataset.  
3. **Evaluate model performance** (loss & perplexity).  
4. **Deploy chatbot** using Flask.

---

## 🏗️ Model Architecture
The fine-tuned GPT-2 uses:
- **Masked multi-head self-attention**
- **Causal masking** (autoregressive text generation)
- **Feedforward neural networks** with GeLU activation
- **Layer normalization** & **residual connections**
- **Positional embeddings** (instead of sinusoidal)

Model chosen: **`gpt2-medium`**  
Final Results:
- **Training Loss:** 0.0306  
- **Validation Loss:** 0.0310  
- **Perplexity:** ~1.03 (good generation performance)

---

## 📂 Dataset
- **Custom-built conversational dataset** (`Question`, `Response` columns)
- Stored in CSV format
- Split: **80% training**, **20% validation**

---

## 🔧 Fine-Tuning Process
1. **Preprocessing**: Clean & split data, load with Hugging Face `datasets`  
2. **Tokenization**: Combine Q&A pairs → max length 128, padded & truncated  
3. **Training**:  
   - Epochs: **3**  
   - Learning Rate: **3e-5**  
   - Batch Size: **2** (for better generalization)  
   - Weight Decay: **0.15** (regularization)  
4. **Evaluation**: Calculated loss & perplexity on validation set  
5. **Saving**: Model & tokenizer saved for deployment

---

## 🚀 Deployment (Flask Web App)
The chatbot runs locally with a simple web interface.

### Run Locally:
```bash
# 1. Install dependencies
pip install torch transformers flask flask-cors datasets

# 2. Start Flask server
python app.py

# 3. Open in browser
http://127.0.0.1:5000
