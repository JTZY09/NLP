from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

MODEL_PATH = "coc_model\\model\\results\\checkpoint-18360"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model & Tokenizer
try:
    print(f" Loading model from {MODEL_PATH}")

    try:
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(" Loaded tokenizer from checkpoint")
    except Exception as e:
        print(f"⚠ Couldn't load tokenizer: {e}")
        print("⚠ Falling back to base GPT-2 tokenizer")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        print(" Loaded model from checkpoint")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("⚠ Trying GPT2LMHeadModel instead")
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)

    model.eval()
    print(" Model successfully loaded!")
    
    # Create text-generation pipeline 
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

except Exception as e:
    print(f" Error loading model: {e}")
    model, tokenizer, generator = None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if generator is None:
        return jsonify({'response': 'Model failed to load. Check server logs.'}), 500

    data = request.json
    message = data.get('message', '').strip()

    try:
        # Generate response using pipeline
        results = generator(message, max_length=100, temperature=0.7, top_k=50, top_p=0.95)

        # Extract generated response
        response = results[0]["generated_text"]

        # Ensure output is trimmed properly
        if response.startswith(message):
            response = response[len(message):].strip()

        return jsonify({'response': response})

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return jsonify({'response': f'Error generating response: {str(e)}'}), 500

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)

    with open("templates/index.html", "w") as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 COC Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: Arial, sans-serif; }
        body { background-color: #f5f5f5; display: flex; align-items: center; justify-content: center; height: 100vh; }
        .chat-container { width: 100%; max-width: 500px; height: 90vh; background: #fff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); display: flex; flex-direction: column; }
        .chat-header { background: #007AFF; color: white; padding: 15px; font-size: 18px; text-align: center; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 70%; padding: 10px; border-radius: 10px; word-wrap: break-word; }
        .message.user { align-self: flex-end; background: #007AFF; color: white; }
        .message.bot { align-self: flex-start; background: #e9e9eb; color: black; }
        .chat-input { display: flex; padding: 15px; border-top: 1px solid #ddd; background: #f5f5f5; }
        .chat-input input { flex: 1; padding: 10px; border-radius: 20px; border: 1px solid #ccc; }
        .send-button { margin-left: 10px; padding: 10px 15px; border-radius: 20px; background: #007AFF; color: white; cursor: pointer; }
        .send-button:hover { background: #005ecb; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">GPT-2 COC Chatbot</div>
        <div class="chat-messages" id="chat-messages"></div>
        <div class="chat-input">
            <input type="text" id="message-input" placeholder="Type a message...">
            <div class="send-button" id="send-button">Send</div>
        </div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const chatMessages = document.getElementById('chat-messages');
            const messageInput = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');
            const API_ENDPOINT = '/generate';

            sendButton.addEventListener('click', sendMessage);
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;

                addMessage(message, 'user');
                messageInput.value = '';

                fetch(API_ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message }),
                })
                .then(response => response.json())
                .then(data => addMessage(data.response, 'bot'))
                .catch(error => console.error('Error:', error));
            }

            function addMessage(text, sender) {
                const messageElement = document.createElement('div');
                messageElement.classList.add('message', sender);
                messageElement.textContent = text;
                chatMessages.appendChild(messageElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        });
    </script>
</body>
</html>""")

    print("✅ Server is starting... Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
