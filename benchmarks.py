import time
import asyncio
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# =====================
# ĐÁNH GIÁ TEXT GENERATION
# =====================

def evaluate_texts(outputs, references):
    bleu_scores, rouge1_scores, rougeL_scores = [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for output, ref in zip(outputs, references):
        bleu = sentence_bleu([ref.split()], output.split())
        rouge = scorer.score(ref, output)
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge['rouge1'].fmeasure)
        rougeL_scores.append(rouge['rougeL'].fmeasure)

    return {
        "BLEU_avg": sum(bleu_scores) / len(bleu_scores),
        "ROUGE-1_avg": sum(rouge1_scores) / len(rouge1_scores),
        "ROUGE-L_avg": sum(rougeL_scores) / len(rougeL_scores),
    }

# =====================
# ĐÁNH GIÁ LABELING
# =====================

def evaluate_labels(preds, truths):
    return {
        "Accuracy": accuracy_score(truths, preds),
        "F1": f1_score(truths, preds, average='macro')
    }

# =====================
# BENCHMARK MODEL
# =====================

async def benchmark_model(model_name, model_fn, task_type, inputs, references):
    """
    model_fn(prompt: str) -> str
    """
    outputs = []
    times = []

    for prompt in tqdm(inputs, desc=f"{model_name} - {task_type}"):
        start = time.time()
        output = await model_fn(prompt)
        duration = time.time() - start
        outputs.append(output.strip())
        times.append(duration)

    avg_time = sum(times) / len(times)

    if task_type == "analysis_explanation":
        metrics = evaluate_texts(outputs, references)
    elif task_type == "labeling":
        metrics = evaluate_labels(outputs, references)
    else:
        raise ValueError("Unknown task type")

    metrics["Avg_Time(s)"] = avg_time
    metrics["Model"] = model_name
    metrics["Task"] = task_type
    return metrics

# =====================
# HÀM MAIN CHẠY CẢ 3 MODEL
# =====================

async def run_benchmark():
    # -------- LOAD DATA --------
    def read_lines(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    analysis_inputs = read_lines("dataset/analysis_explanation/input.txt")
    analysis_refs   = read_lines("dataset/analysis_explanation/reference.txt")

    label_inputs = read_lines("dataset/labeling/input.txt")
    label_refs   = read_lines("dataset/labeling/label_true.txt")

    # -------- ĐỊNH NGHĨA 3 MODEL WRAPPER --------
    # Bạn sẽ tự hiện thực các hàm infer_* dùng openai client của bạn
    async def infer_qwen(prompt): ...
    async def infer_gpt(prompt): ...
    async def infer_gemini(prompt): ...

    models = {
        "Qwen3-14B": infer_qwen,
        "ChatGPT": infer_gpt,
        "Gemini": infer_gemini
    }

    results = []

    # -------- CHẠY BENCHMARK --------
    for name, fn in models.items():
        res1 = await benchmark_model(name, fn, "analysis_explanation", analysis_inputs, analysis_refs)
        res2 = await benchmark_model(name, fn, "labeling", label_inputs, label_refs)
        results.extend([res1, res2])

    # -------- LƯU KẾT QUẢ --------
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\n=== Kết quả tổng hợp ===")
    print(df)

# =====================
# CHẠY
# =====================

if __name__ == "__main__":
    asyncio.run(run_benchmark())
