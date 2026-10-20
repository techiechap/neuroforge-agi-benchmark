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
    import json
    from pathlib import Path
    from src.benchmark_quality import extract_pass

    llm = OllamaLLM()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_file = Path(f"neuroforge_results_{stamp}.csv")
    stats = {"SUCCESS": 0, "FAILURE": 0, "ERROR": 0}
    with results_file.open("x", newline="") as f:
        csv.writer(f).writerow(["Trial", "Timestamp_UTC", "Seed", "Status", "Error_Type"])
    for i in range(1, iterations + 1):
        os.environ["NEUROFORGE_CASE_SEED"] = str(i)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        error_type = ""
        try:
            run = spatial_inhibition_task.run(llm=llm)
            status = "SUCCESS" if extract_pass(run) else "FAILURE"
        except Exception as exc:
            status = "ERROR"
            error_type = type(exc).__name__
        stats[status] += 1
        with results_file.open("a", newline="") as f:
            csv.writer(f).writerow([i, now, i, status, error_type])
    total = sum(stats.values())
    print(json.dumps({"csv": str(results_file), "requested": iterations,
                      "recorded": total, "counts": stats}, indent=2))
    if total != iterations or stats["ERROR"]:
        raise RuntimeError("Benchmark incomplete or had execution errors; see CSV")

if __name__ == "__main__":
    run_final_benchmark(iterations=20)
