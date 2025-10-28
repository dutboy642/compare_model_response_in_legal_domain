import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
def run_parallel_tasks_with_saving(
    jobs,
    job_fn,
    output_csv_path,
    columns,
    max_workers=5,
    save_every=10,
    price_column="price"
):
    """
    Hàm chạy song song (generic) cho các tác vụ LLM hoặc API I/O bound.

    Args:
        jobs: list các tuple chứa input cho job_fn
        job_fn: hàm nhận *args -> trả về dict (một dòng dữ liệu)
        output_csv_path: đường dẫn file CSV để lưu
        columns: danh sách các cột (đảm bảo thứ tự khi khởi tạo CSV)
        max_workers: số luồng song song
        save_every: số job sau đó sẽ flush tạm vào file
        price_column: tên cột để tính tổng chi phí

    Returns:
        pandas.DataFrame: DataFrame kết quả cuối cùng
    """

    # Load file cũ nếu có
    if os.path.exists(output_csv_path):
        results_df = pd.read_csv(output_csv_path)
        print(f"📂 Loaded {len(results_df)} existing rows from {output_csv_path}")
    else:
        results_df = pd.DataFrame(columns=columns)

    results = []

    def calculate_total_price(df_or_path):
        """Hàm nhỏ nội bộ để tính tổng tiền"""
        if isinstance(df_or_path, str) and os.path.exists(df_or_path):
            df = pd.read_csv(df_or_path)
        else:
            df = df_or_path
        if price_column not in df.columns:
            return 0
        return df[price_column].fillna(0).sum()

    # ---- Run song song ----
    print(f"🚀 Running {len(jobs)} jobs with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(job_fn, *job): job for job in jobs}

        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            # Flush tạm
            if count % save_every == 0:
                temp_df = pd.DataFrame(results)
                results_df = pd.concat([results_df, temp_df], ignore_index=True)
                results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
                total_price = calculate_total_price(results_df)
                print(f"💾 Saved {len(results_df)} rows (processed {count}/{len(jobs)}) | 💰 Total = {total_price:.4f}")
                results.clear()

    # ---- Flush phần còn lại ----
    if results:
        temp_df = pd.DataFrame(results)
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        total_price = calculate_total_price(results_df)
        print(f"💾 Final save ({len(results_df)} rows) | 💰 Total = {total_price:.4f}")

    print(f"✅ Completed. Total rows: {len(results_df)} saved to {output_csv_path}")
    return results_df