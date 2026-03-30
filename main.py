import os
import csv
import datetime
import requests
from src.agi_benchmark import spatial_inhibition_task

# 1. Environment Stability
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class OllamaLLM:
    def __init__(self, model="gemma3", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.url = f"{self.host}/api/chat"

    def prompt(self, message: str) -> str:
        resp = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": message}],
                "stream": False,
                "options": {"temperature": 0.1} # Low temp for consistency
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

def run_final_benchmark(iterations=20):
    llm = OllamaLLM()
    results_file = "neuroforge_results.csv"
    
    stats = {"SUCCESS": 0, "FAILURE": 0}
    failure_log = []

    print(f"🛠️ Starting Neuroforge Benchmark: {iterations} Iterations")
    print(f"📊 Logging to: {results_file}\n")

    # Prepare CSV Header
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Trial", "Timestamp", "Status", "Grid_Size", "Error_Type"])

    for i in range(1, iterations + 1):
        timestamp = datetime.datetime.now().isoformat()
        print(f"🔄 Trial {i:02d}/{iterations}...", end=" ", flush=True)
        
        try:
            # The task returns True/False and prints its own debug info
            is_correct = spatial_inhibition_task.run(llm=llm)
            
            # Note: spatial_inhibition_task.run returns a kbench.Run object
            # We access the actual result from the run object
            passed = is_correct.result if hasattr(is_correct, 'result') else is_correct
            
            status = "SUCCESS" if passed else "FAILURE"
            stats[status] += 1
            
            # Append to CSV
            with open(results_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i, timestamp, status, "Dynamic", "N/A" if passed else "Logic_Error"])
            
            icon = "✅" if passed else "❌"
            print(f"{icon}")

        except Exception as e:
            print(f"⚠️ CRASH: {e}")

    # --- FINAL REPORT ---
    total = stats["SUCCESS"] + stats["FAILURE"]
    success_rate = (stats["SUCCESS"] / total) * 100 if total > 0 else 0
    
    print("\n" + "="*40)
    print("🏆 FINAL BENCHMARK REPORT")
    print("="*40)
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Passes: {stats['SUCCESS']}")
    print(f"Total Fails:  {stats['FAILURE']}")
    print("="*40)
    print(f"👉 Upload {results_file} to your Kaggle Writeup!")

if __name__ == "__main__":
    run_final_benchmark(iterations=20)