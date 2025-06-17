import os, requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
# Optionally set these for the new services:
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

print(f"xAI API key loaded: {bool(GROK_API_KEY)}")

if not all([ANTHROPIC_API_KEY, GEMINI_API_KEY, PERPLEXITY_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, HUGGINGFACE_API_KEY, GROK_API_KEY]):
    raise RuntimeError("Missing keys in .env")

	
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory('.', '01--index.html')

@app.route("/api/anthropic", methods=["POST"])
def anthropic_complete():
    # Use the Messages API
    headers = {
      "Content-Type": "application/json",
      "x-api-key": ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01"
    }
    payload = {
      "model": "claude-3-5-sonnet-20240620",
      "max_tokens": 1024,
      "messages": [
        {"role": "user", "content": request.json.get("prompt","")}
      ]
    }
    resp = requests.post("https://api.anthropic.com/v1/messages",
                         json=payload, headers=headers)
    try:
        resp.raise_for_status()
        return jsonify(resp.json()), resp.status_code
    except requests.HTTPError:
        return jsonify({
          "error": "Anthropic error",
          "status_code": resp.status_code,
          "body": resp.text
        }), resp.status_code

@app.route("/api/gemini", methods=["POST"])
def gemini_complete():
    prompt = request.json.get("prompt", "")
    model_name = "gemini-1.5-pro-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 300}
    }

    resp = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload, headers={"Content-Type": "application/json"})
    try:
        resp.raise_for_status()
        data = resp.json()
        # Normalize Gemini's response into content blocks
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        blocks = [{"type": "text", "text": part.get("text", "")} for part in parts]
        full = "".join([b["text"] for b in blocks])
        return jsonify({"content": blocks, "completion": full}), resp.status_code
    except requests.HTTPError:
        return jsonify({"error": "Gemini error", "status_code": resp.status_code, "body": resp.text}), resp.status_code

@app.route("/api/perplexity", methods=["POST"])
def perplexity_complete():
    prompt = request.json.get("prompt", "")
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "sonar-pro",               # or "sonar" if you prefer :contentReference[oaicite:0]{index=0}
        "messages": [
            { "role": "user", "content": prompt }
        ]
    }
    resp = requests.post(url, headers=headers, json=body)
    try:
        resp.raise_for_status()
        data = resp.json()
        # extract answer text
        text = data["choices"][0]["message"]["content"]
        return jsonify({
            "content": [{"type": "text", "text": text}],
            "completion": text
        }), 200
    except requests.HTTPError:
        return jsonify({
            "error": "Perplexity API error",
            "status_code": resp.status_code,
            "body": resp.text
        }), resp.status_code
    return jsonify({
        "error": "Perplexity endpoint not implemented"
    }), 501

@app.route("/api/deepseek", methods=["POST"])
def deepseek_complete():
    prompt = request.json.get("prompt", "")
    # Use official DeepSeek chat completions endpoint
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization":f"Bearer {DEEPSEEK_API_KEY}","Content-Type":"application/json"}
    body = {"model":"deepseek-chat","messages":[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":prompt}
            ],
            "frequency_penalty":0
    }
    resp = requests.post(url, headers=headers, json=body)
    try:
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return jsonify({"content":[{"type":"text","text":text}],"completion":text}), 200
    except requests.HTTPError:
        # Log actual response for debugging
        print(f"[DeepSeek ERROR] {resp.status_code}: {resp.text}")
        return jsonify({"error":"DeepSeek API error","status_code":resp.status_code,"body":resp.text}), resp.status_code

    

@app.route("/api/openai", methods=["POST"])
def openai_complete():
    prompt = request.json.get("prompt", "")
    # Example OpenAI integration:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content
        # wrap in same shape
        return jsonify({
            "content": [{"type": "text", "text": text}],
            "completion": text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/llama", methods=["POST"])
def llama_complete():
    prompt = request.json.get("prompt", "")
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"       # OPEN SOURCE MODEL
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    # Zephyr uses ChatML format
    formatted_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    body = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        print(f"[Zephyr] Response: {resp.status_code}")
        
        if resp.status_code == 503:
            return jsonify({
                "error": "Model is loading, please try again in 20-30 seconds",
                "status_code": 503
            }), 503
            
        resp.raise_for_status()
        data = resp.json()
        
        # Handle both list and dict responses
        if isinstance(data, list):
            generated_text = data[0].get("generated_text", "")
        else:
            generated_text = data.get("generated_text", "")
            
        # Clean any remaining tags
        clean_text = generated_text.replace("<|assistant|>", "").strip()
        
        return jsonify({
            "content": [{"type": "text", "text": clean_text}],
            "completion": clean_text
        }), 200
        
    except Exception as e:
        error = f"Zephyr error: {str(e)}"
        if hasattr(e, 'response'):
            error += f" (Status: {e.response.status_code}, Response: {e.response.text[:200]})"
        return jsonify({"error": error}), 500

		
@app.route("/api/grok", methods=["POST"])
def grok_complete():
    prompt = request.json.get("prompt", "")
    url = "https://api.x.ai/v1/chat/completions"  # Chat completions endpoint :contentReference[oaicite:0]{index=0}
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "grok-3-beta",                      # Use a valid Grok model name :contentReference[oaicite:1]{index=1}
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    resp = requests.post(url, headers=headers, json=payload)
    try:
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return jsonify({
            "content": [{"type": "text", "text": text}],
            "completion": text
        }), 200
    except requests.HTTPError:
        return jsonify({
            "error": "Grok API error",
            "status_code": resp.status_code,
            "body": resp.text
        }), resp.status_code


    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)), debug=True)