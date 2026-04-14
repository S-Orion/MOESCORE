import sys
import os
import pandas as pd
from math import sqrt
from typing import Iterable, List, Tuple
import csv
# import utils.utils as utils

def _check_same_length(x: Iterable[float], y: Iterable[float]) -> Tuple[List[float], List[float]]:
    x = list(x)
    y = list(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least 2 samples are required.")
    return x, y

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals)

def _ranks(values: List[float]) -> List[float]:
    """
    Assign ranks to values. Ties are handled by assigning the average rank.
    Example: [30, 10, 20, 20] -> [4.0, 1.0, 2.5, 2.5]
    """
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i + 1
        # Find the range of tied values
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        # i..j-1 is a tie group. Average rank = (i+1 + j)/2
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            idx = indexed[k][0]
            ranks[idx] = avg_rank
        i = j
    return ranks

def lcc_pearson(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Pearson's correlation coefficient (Linear Correlation Coefficient, LCC).
    Returns 0.0 if one sequence has zero variance.
    """
    x, y = _check_same_length(x, y)
    mx, my = _mean(x), _mean(y)
    sxx = syy = sxy = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        dy = yi - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    return sxy / sqrt(sxx * syy)

def srcc_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Spearman Rank Correlation Coefficient (SRCC).
    Compute ranks with tie handling, then apply Pearson correlation.
    """
    x, y = _check_same_length(x, y)
    rx = _ranks(x)
    ry = _ranks(y)
    return lcc_pearson(rx, ry)

def _tie_pairs_count(values: List[float]) -> int:
    """
    Count the number of tied pairs in the sequence:
    sum over groups of comb(t_i, 2).
    """
    from collections import Counter
    c = Counter(values)
    ties = 0
    for t in c.values():
        if t >= 2:
            ties += t * (t - 1) // 2
    return ties

def ktau_b(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Kendall's tau-b (tie corrected).
    tau-b = (C - D) / sqrt((n0 - n1) * (n0 - n2))
      - n0 = total number of pairs = n*(n-1)/2
      - n1 = number of tied pairs in x
      - n2 = number of tied pairs in y
      - C  = concordant pairs
      - D  = discordant pairs
    Ties in x or y are excluded from C and D.
    """
    x, y = _check_same_length(x, y)
    n = len(x)
    n0 = n * (n - 1) // 2
    if n0 == 0:
        return 0.0

    # Count concordant and discordant pairs (O(n^2))
    C = D = 0
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            if dx == 0 or dy == 0:
                # skip ties
                continue
            prod = dx * dy
            if prod > 0:
                C += 1
            elif prod < 0:
                D += 1

    n1 = _tie_pairs_count(list(x))
    n2 = _tie_pairs_count(list(y))

    denom = sqrt((n0 - n1) * (n0 - n2))
    if denom == 0.0:
        return 0.0
    return (C - D) / denom

def mse(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Mean Squared Error (MSE).
    """
    x, y = _check_same_length(x, y)
    s = 0.0
    for xi, yi in zip(x, y):
        d = xi - yi
        s += d * d
    return s / len(x)

def evaluate_all(pred, gt):
    # optional: basic sanitization
    pairs = [(float(a), float(b)) for a, b in zip(pred, gt) if (a == a and b == b)]  # drop NaN
    if len(pairs) < 2:
        raise ValueError("Need at least 2 valid pairs after filtering.")
    xp, yg = zip(*pairs)
    return {
        "SRCC": srcc_spearman(xp, yg),
        "LCC" : lcc_pearson(xp, yg),
        "KTAU": ktau_b(xp, yg),
        "MSE" : mse(xp, yg),
        "N"   : len(pairs),
    }

def main(result_csv_path, valid_list_path, save_csv_path):
    df_pred = pd.read_csv(result_csv_path)
    df_ref  = pd.read_csv(valid_list_path, usecols=["wav_file_name", "average_score"])
    df_cat  = pd.merge(df_pred, df_ref, on="wav_file_name", how="inner")
    pred = df_cat["pred_score"].tolist()
    gt   = df_cat["average_score"].tolist()

    result = evaluate_all(pred, gt)
    with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ["SRCC", "LCC", "KTAU", "MSE", "N"]:
            w.writerow([k, result[k]])
    
    return

if __name__ == "__main__":
    # Usage: python evaluate.py <inference_csv_path> <validation_list_path> <save_dir>
    if len(sys.argv) < 4:
        print("Usage: python evaluate.py <inference_csv_path> <validation_csv_path> <save_dir>")
        sys.exit(1)

    inference_csv_path   = sys.argv[1]
    validation_list_path = sys.argv[2]
    save_dir             = sys.argv[3]

    # Validate inputs
    if not os.path.isfile(inference_csv_path):
        print(f"Error: Inference CSV not found: {inference_csv_path}")
        sys.exit(1)

    if not os.path.isfile(validation_list_path):
        print(f"Error: Validation list not found: {validation_list_path}")
        sys.exit(1)

    # Ensure save directory exists (create if missing)
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Failed to create save directory '{save_dir}': {e}")
        sys.exit(1)

    # Fixed output filename inside the specified save directory
    save_csv_path = os.path.join(save_dir, "evaluation_result.csv")

    # Logging
    print("The evaluation will be conducted based on the following two files:")
    print(f"\t- {os.path.abspath(inference_csv_path)}")
    print(f"\t- {os.path.abspath(validation_list_path)}")
    print(f"The result will be saved to: {os.path.abspath(save_csv_path)}")

    # Call main with three arguments (CSV paths + fixed save file path)
    main(inference_csv_path, validation_list_path, save_csv_path)
