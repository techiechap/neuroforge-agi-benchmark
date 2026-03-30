import os
import sys
import kaggle_benchmarks as kbench

# FIX: Force the math libraries to play nice before any logic loads
# 1. FIX MATH CONFLICTS FIRST
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# OPTIONAL: If the iJIT error persists, this line bypasses the check
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 1. Credentials
os.environ['KAGGLE_USERNAME'] = "ashishjd"
os.environ['KAGGLE_KEY'] = "KGAT_6bf474790adabbfbe37c0f6198297a9d"
import requests

class OllamaLLM:
    def __init__(self, model="gemma3", host="http://localhost:11434"):
        self.model = model
        self.url = f"{host.rstrip('/')}/api/chat"

    def prompt(self, message: str) -> str:
        resp = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": message}
                ],
                "stream": False,
                "keep_alive": "10m"
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


llm = OllamaLLM(model="gemma3")

question = "Explain RNA folding in two lines."
answer = llm.prompt(question)
print(answer)

def run_local_validation():
    print("🛠️  Neuroforge AGI: Handshaking with Gemini Proxy...")
    
    try:
        from src.agi_benchmark import spatial_inhibition_task
        
        # 3. THE FINAL FIX: 
        # In this SDK version, LLMChat takes the model string as the name.
        model_id = "google/gemini-2.0-flash"
        print(f"🔗 Connecting to {model_id}...")
        
        # Based on your dir(kbench), LLMChat is top-level.
        # We pass the model_id as the name of the actor.
        # llm = kbench.LLMChat
        llm = OllamaLLM(model="gemma3")
        print(f"🧪 Running Task: {spatial_inhibition_task.name}")

        # 4. RUN
        result = spatial_inhibition_task.run(llm=llm)
        
        print("\n" + "⭐" * 30)
        print(f"✅ SUCCESS!")
        # Safety check for score access
        score_val = getattr(result.score, 'value', result.score)
        print(f"SCORE: {score_val * 100}%")
        print("⭐" * 30)

    except Exception as e:
        print(f"❌ Execution Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_local_validation()