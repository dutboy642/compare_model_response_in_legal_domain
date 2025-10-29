import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import json


def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score for a single reference-candidate pair
    """
    if pd.isna(reference) or pd.isna(candidate) or reference == '' or candidate == '':
        return 0.0
    
    # Tokenize by characters for Japanese text
    reference_tokens = [list(str(reference))]
    candidate_tokens = list(str(candidate))
    
    # Use smoothing function to handle edge cases
    smoothie = SmoothingFunction().method4
    
    try:
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
        return score
    except:
        return 0.0


def calculate_rouge_scores(reference, candidate):
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    """
    if pd.isna(reference) or pd.isna(candidate) or reference == '' or candidate == '':
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(str(reference), str(candidate))
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def calculate_label_metrics(true_labels, pred_labels):
    """
    Calculate accuracy, precision, recall, and F1 score for label classification
    """
    # Remove any NaN values
    mask = ~(pd.isna(true_labels) | pd.isna(pred_labels))
    true_labels_clean = true_labels[mask]
    pred_labels_clean = pred_labels[mask]
    
    if len(true_labels_clean) == 0:
        return {
            'accuracy': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'weighted_precision': 0.0,
            'weighted_recall': 0.0,
            'weighted_f1': 0.0
        }
    
    metrics = {
        'accuracy': accuracy_score(true_labels_clean, pred_labels_clean),
        'macro_precision': precision_score(true_labels_clean, pred_labels_clean, average='macro', zero_division=0),
        'macro_recall': recall_score(true_labels_clean, pred_labels_clean, average='macro', zero_division=0),
        'macro_f1': f1_score(true_labels_clean, pred_labels_clean, average='macro', zero_division=0),
        'weighted_precision': precision_score(true_labels_clean, pred_labels_clean, average='weighted', zero_division=0),
        'weighted_recall': recall_score(true_labels_clean, pred_labels_clean, average='weighted', zero_division=0),
        'weighted_f1': f1_score(true_labels_clean, pred_labels_clean, average='weighted', zero_division=0)
    }
    
    return metrics


def evaluate_file(file_path):
    """
    Evaluate a single CSV file
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Check required columns
    required_cols = ['explanation_ja', 'generated_explanation', 'label', 'generated_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return None
    
    print(f"\nTotal rows: {len(df)}")
    
    # Calculate BLEU scores
    print("\nCalculating BLEU scores...")
    bleu_scores = []
    for idx, row in df.iterrows():
        score = calculate_bleu_score(row['explanation_ja'], row['generated_explanation'])
        bleu_scores.append(score)
    
    avg_bleu = np.mean(bleu_scores)
    
    # Calculate ROUGE scores
    print("Calculating ROUGE scores...")
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx, row in df.iterrows():
        scores = calculate_rouge_scores(row['explanation_ja'], row['generated_explanation'])
        rouge1_scores.append(scores['rouge1'])
        rouge2_scores.append(scores['rouge2'])
        rougeL_scores.append(scores['rougeL'])
    
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougeL = np.mean(rougeL_scores)
    
    # Calculate label metrics
    print("Calculating label classification metrics...")
    label_metrics = calculate_label_metrics(df['label'], df['generated_label'])
    
    # Calculate cost and time metrics
    cost_time_metrics = {}
    if 'price' in df.columns:
        # Remove NaN values for price calculation
        valid_prices = df['price'][~pd.isna(df['price'])]
        cost_time_metrics['total_price'] = float(valid_prices.sum()) if len(valid_prices) > 0 else 0.0
        cost_time_metrics['avg_price'] = float(valid_prices.mean()) if len(valid_prices) > 0 else 0.0
        cost_time_metrics['min_price'] = float(valid_prices.min()) if len(valid_prices) > 0 else 0.0
        cost_time_metrics['max_price'] = float(valid_prices.max()) if len(valid_prices) > 0 else 0.0
    
    if 'generation_time' in df.columns:
        # Remove NaN values for time calculation
        valid_times = df['generation_time'][~pd.isna(df['generation_time'])]
        cost_time_metrics['total_time'] = float(valid_times.sum()) if len(valid_times) > 0 else 0.0
        cost_time_metrics['avg_time'] = float(valid_times.mean()) if len(valid_times) > 0 else 0.0
        cost_time_metrics['min_time'] = float(valid_times.min()) if len(valid_times) > 0 else 0.0
        cost_time_metrics['max_time'] = float(valid_times.max()) if len(valid_times) > 0 else 0.0
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\n--- Text Generation Metrics (explanation_ja vs generated_explanation) ---")
    print(f"BLEU Score:        {avg_bleu:.4f}")
    print(f"ROUGE-1 F1:        {avg_rouge1:.4f}")
    print(f"ROUGE-2 F1:        {avg_rouge2:.4f}")
    print(f"ROUGE-L F1:        {avg_rougeL:.4f}")
    
    print("\n--- Label Classification Metrics (label vs generated_label) ---")
    print(f"Accuracy:          {label_metrics['accuracy']:.4f}")
    print(f"Macro Precision:   {label_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:      {label_metrics['macro_recall']:.4f}")
    print(f"Macro F1:          {label_metrics['macro_f1']:.4f}")
    print(f"Weighted Precision:{label_metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:   {label_metrics['weighted_recall']:.4f}")
    print(f"Weighted F1:       {label_metrics['weighted_f1']:.4f}")
    
    # Print cost and time metrics if available
    if cost_time_metrics:
        print("\n--- Cost and Time Metrics ---")
        if 'total_price' in cost_time_metrics:
            print(f"Total Price:       ${cost_time_metrics['total_price']:.6f}")
            print(f"Average Price:     ${cost_time_metrics['avg_price']:.6f}")
            print(f"Min Price:         ${cost_time_metrics['min_price']:.6f}")
            print(f"Max Price:         ${cost_time_metrics['max_price']:.6f}")
        if 'total_time' in cost_time_metrics:
            print(f"Total Time:        {cost_time_metrics['total_time']:.2f}s")
            print(f"Average Time:      {cost_time_metrics['avg_time']:.2f}s")
            print(f"Min Time:          {cost_time_metrics['min_time']:.2f}s")
            print(f"Max Time:          {cost_time_metrics['max_time']:.2f}s")
    
    # Print classification report
    print("\n--- Detailed Classification Report ---")
    mask = ~(pd.isna(df['label']) | pd.isna(df['generated_label']))
    if mask.sum() > 0:
        print(classification_report(df['label'][mask], df['generated_label'][mask], zero_division=0))
    
    # Compile results
    results = {
        'file': os.path.basename(file_path),
        'total_rows': len(df),
        'text_metrics': {
            'bleu': float(avg_bleu),
            'rouge1': float(avg_rouge1),
            'rouge2': float(avg_rouge2),
            'rougeL': float(avg_rougeL)
        },
        'label_metrics': {k: float(v) for k, v in label_metrics.items()},
        'cost_time_metrics': cost_time_metrics
    }
    
    return results


def evaluate_all_files(directory):
    """
    Evaluate all CSV files in the model_response directory
    """
    all_results = []
    
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files to evaluate")
    
    for csv_file in sorted(csv_files):
        file_path = os.path.join(directory, csv_file)
        result = evaluate_file(file_path)
        if result:
            all_results.append(result)
    
    # Save results to JSON
    output_file = os.path.join(directory, 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print("\n{:<40} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Model", "BLEU", "ROUGE-L", "Accuracy", "Macro-F1", "Weighted-F1"))
    print("-" * 80)
    
    for result in all_results:
        model_name = result['file'].replace('_response.csv', '')
        print("{:<40} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
            model_name,
            result['text_metrics']['bleu'],
            result['text_metrics']['rougeL'],
            result['label_metrics']['accuracy'],
            result['label_metrics']['macro_f1'],
            result['label_metrics']['weighted_f1']
        ))
    
    # Print cost and time comparison if available
    has_price = any('total_price' in r.get('cost_time_metrics', {}) for r in all_results)
    has_time = any('total_time' in r.get('cost_time_metrics', {}) for r in all_results)
    
    if has_price or has_time:
        print("\n" + "="*80)
        print("COST AND TIME COMPARISON")
        print("="*80)
        
        if has_price and has_time:
            print("\n{:<40} {:>12} {:>12} {:>12} {:>12}".format(
                "Model", "Total Price", "Avg Price", "Total Time", "Avg Time"))
            print("-" * 80)
            for result in all_results:
                model_name = result['file'].replace('_response.csv', '')
                ctm = result.get('cost_time_metrics', {})
                print("{:<40} ${:>11.6f} ${:>11.6f} {:>11.2f}s {:>11.2f}s".format(
                    model_name,
                    ctm.get('total_price', 0.0),
                    ctm.get('avg_price', 0.0),
                    ctm.get('total_time', 0.0),
                    ctm.get('avg_time', 0.0)
                ))
        elif has_price:
            print("\n{:<40} {:>12} {:>12}".format("Model", "Total Price", "Avg Price"))
            print("-" * 80)
            for result in all_results:
                model_name = result['file'].replace('_response.csv', '')
                ctm = result.get('cost_time_metrics', {})
                print("{:<40} ${:>11.6f} ${:>11.6f}".format(
                    model_name,
                    ctm.get('total_price', 0.0),
                    ctm.get('avg_price', 0.0)
                ))
        elif has_time:
            print("\n{:<40} {:>12} {:>12}".format("Model", "Total Time", "Avg Time"))
            print("-" * 80)
            for result in all_results:
                model_name = result['file'].replace('_response.csv', '')
                ctm = result.get('cost_time_metrics', {})
                print("{:<40} {:>11.2f}s {:>11.2f}s".format(
                    model_name,
                    ctm.get('total_time', 0.0),
                    ctm.get('avg_time', 0.0)
                ))


if __name__ == "__main__":
    # Directory containing the CSV files
    model_response_dir = "/home/thanhnguyen/code/compare_model_response_in_legal_domain/model_response"
    
    # Evaluate all files
    evaluate_all_files(model_response_dir)
