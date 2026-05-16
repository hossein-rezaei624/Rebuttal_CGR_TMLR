"""
Analysis script for CGR diagnostic logs produced by cgr_with_diag.py
(when run with --cgr_diag_log).

Computes ACROSS ALL SEEDS:
  (b.1) Cross-seed Spearman rank correlation of per-sample variance vectors
        --> mean ± std over the 10 pairs of seeds.
  (a.2) Variance vs forgetting-event Spearman correlation
        --> computed separately for EACH seed; reported as mean ± std over seeds.
  (b.2) Diagnostic table comparing CGR vs random / high-loss / low-confidence
        --> each cell computed separately for EACH seed; reported as mean ± std.

Usage:
    python analyze_cgr_diag.py --diag_dir cgr_diag_logs --E 4 --buffer_size 1000
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


# ----------------------------- I/O -----------------------------

def load_seed_logs(diag_dir):
    paths = sorted(Path(diag_dir).glob('cgr_diag_seed*.pt'))
    if not paths:
        raise FileNotFoundError(f"No 'cgr_diag_seed*.pt' files found in {diag_dir}")
    return [torch.load(p, map_location='cpu') for p in paths]


# ------------------------- Metrics ---------------------------

def variance_from_eval_confidence(log, E):
    """CGR's actual variance signal: variance of eval-mode confidence over first E epochs."""
    conf = log['cgr_confidence_by_sample']
    return conf[:E].var(dim=0).numpy()


def forgetting_events(correct):
    """Toneva-style forgetting events: # of correct -> incorrect transitions over training."""
    correct = correct.bool()
    transitions = correct[:-1] & ~correct[1:]
    return transitions.sum(dim=0).numpy()


# --------------------- (b.1) Cross-seed -----------------------

def cross_seed_spearman(logs, E):
    variances = [variance_from_eval_confidence(log, E) for log in logs]
    lens = {v.shape[0] for v in variances}
    if len(lens) != 1:
        raise ValueError(f"Variance vectors differ across seeds: {lens}")

    rhos = []
    pairs = []
    n = len(variances)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(variances[i], variances[j])
            rhos.append(r)
            pairs.append((logs[i]['seed'], logs[j]['seed']))
    return float(np.mean(rhos)), float(np.std(rhos)), rhos, pairs


# ---------------- (a.2) Variance vs forgetting (all seeds) ----------------

def variance_vs_forgetting_per_seed(logs, E):
    """Compute Spearman ρ between variance and forgetting events for EACH seed."""
    results = []
    for log in logs:
        variance = variance_from_eval_confidence(log, E)
        forgetting = forgetting_events(log['diag_correct'])
        r, p = spearmanr(variance, forgetting)
        results.append({'seed': log['seed'], 'rho': float(r), 'p': float(p)})
    rhos = [r['rho'] for r in results]
    return results, float(np.mean(rhos)), float(np.std(rhos))


# ---------------- (b.2) Diagnostic table (all seeds) ----------------------

def diagnostic_table_one_seed(log, E, buffer_size, last_k_for_margin, random_seed):
    n_epochs = log['diag_target_conf'].shape[0] if 'diag_target_conf' in log else None
    # Note: cgr_with_diag.py saves cgr_confidence_by_sample as the eval-mode target conf,
    # filled for ALL task-1 epochs when --cgr_diag_log is set. Use it for both
    # CGR's variance (first E rows) and the per-epoch confidence trajectory.
    target_conf_all = log['cgr_confidence_by_sample']  # (n_epochs, n_samples)
    n_epochs = target_conf_all.shape[0]
    labels = log['diag_labels'].numpy()

    # Per-sample selection scores (computed over first E epochs, matching CGR's window
    # and Figure 3 sub-figure b.1)
    variance = variance_from_eval_confidence(log, E)
    mean_conf_E = target_conf_all[:E].mean(dim=0).numpy()
    mean_loss_E = log['diag_loss'][:E].mean(dim=0).numpy()

    # Per-sample reporting metrics. Mean target confidence is computed over the
    # same first-E-epoch window as the selection / as Figure 3 (b.1).
    margin_late = log['diag_margin'][-last_k_for_margin:].mean(dim=0).numpy()
    forgetting = forgetting_events(log['diag_correct'])
    mean_conf_report = mean_conf_E  # = target_conf_all[:E].mean(dim=0).numpy()

    # Per-class top-K (K = buffer_size // num_classes seen in task 1)
    unique_classes = np.unique(labels[labels >= 0])
    num_classes = len(unique_classes)
    k_per_class = buffer_size // num_classes

    def top_k_per_class(score, descending=True):
        out = []
        for c in unique_classes:
            idx = np.where(labels == c)[0]
            order = np.argsort(score[idx])
            if descending:
                order = order[::-1]
            out.append(idx[order[:k_per_class]])
        return np.concatenate(out)

    rng = np.random.default_rng(random_seed)
    rules = {
        'Random':          np.concatenate([
                               rng.choice(np.where(labels == c)[0],
                                          size=min(k_per_class, (labels == c).sum()),
                                          replace=False)
                               for c in unique_classes
                           ]),
        'High loss':       top_k_per_class(mean_loss_E, descending=True),
        'High confidence': top_k_per_class(mean_conf_E, descending=True),
        'Low confidence':  top_k_per_class(mean_conf_E, descending=False),
        'CGR (variance)':  top_k_per_class(variance, descending=True),
    }

    row_dict = {}
    for name, idx in rules.items():
        row_dict[name] = {
            'mean_margin': float(margin_late[idx].mean()),
            'mean_forgetting': float(forgetting[idx].mean()),
            'mean_target_conf': float(mean_conf_report[idx].mean()),
        }
    return row_dict, k_per_class, num_classes


def diagnostic_table_all_seeds(logs, E, buffer_size, last_k_for_margin):
    """Compute the diagnostic table per seed, then aggregate to mean ± std."""
    per_seed_rows = []
    k_per_class, num_classes = None, None
    for log in logs:
        # Use the seed itself as the random_seed for the Random rule, so the
        # randomness is reproducible and seed-specific.
        row_dict, k, nc = diagnostic_table_one_seed(
            log, E, buffer_size,
            last_k_for_margin=last_k_for_margin,
            random_seed=int(log['seed']) if str(log['seed']).isdigit() else 0
        )
        per_seed_rows.append(row_dict)
        k_per_class, num_classes = k, nc

    # Aggregate across seeds
    rule_names = list(per_seed_rows[0].keys())
    agg = {}
    for name in rule_names:
        agg[name] = {}
        for metric in ['mean_margin', 'mean_forgetting', 'mean_target_conf']:
            vals = [seed_row[name][metric] for seed_row in per_seed_rows]
            agg[name][metric + '_mean'] = float(np.mean(vals))
            agg[name][metric + '_std']  = float(np.std(vals))
            agg[name][metric + '_per_seed'] = [float(v) for v in vals]
    return agg, per_seed_rows, k_per_class, num_classes


# ------------------------- Reporting -------------------------

def print_b1(mean_rho, std_rho, all_rhos, pairs, n_seeds):
    n_pairs = len(all_rhos)
    print(f"\n=== (b.1) Cross-seed Spearman correlation of variance vectors ===")
    print(f"Number of seeds: {n_seeds}  ({n_pairs} pairs)")
    print(f"Mean ρ ± std: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"Per-pair ρ values:")
    for (s1, s2), r in zip(pairs, all_rhos):
        print(f"  (seed {s1}, seed {s2}): ρ = {r:.4f}")
    print(f"\n  Paper insertion: \\bar\\rho = {mean_rho:.2f} \\pm {std_rho:.2f}")


def print_a2(results, mean_rho, std_rho):
    print(f"\n=== (a.2) Variance vs forgetting events (ALL seeds) ===")
    print(f"Per-seed ρ values:")
    for r in results:
        sig = '***' if r['p'] < 1e-50 else ('**' if r['p'] < 1e-10 else '')
        print(f"  seed {r['seed']}: ρ = {r['rho']:.4f}  (p = {r['p']:.3e})  {sig}")
    print(f"\nMean ρ ± std over {len(results)} seeds: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"\n  Paper insertion: \\rho = {mean_rho:.2f} \\pm {std_rho:.2f}")


def print_b2(agg, per_seed_rows, k_per_class, num_classes, n_seeds):
    print(f"\n=== (b.2) Diagnostic table (averaged over {n_seeds} seeds) ===")
    print(f"Per-class budget K = {k_per_class}  ({num_classes} classes seen in task 1)\n")

    rule_names = list(agg.keys())
    header = f"{'Rule':<18} {'Margin (mean±std)':>22} {'Forget (mean±std)':>22} {'MeanConf (mean±std)':>22}"
    print(header)
    print('-' * len(header))
    for name in rule_names:
        d = agg[name]
        print(f"{name:<18} "
              f"{d['mean_margin_mean']:>7.4f} ± {d['mean_margin_std']:.4f}    "
              f"{d['mean_forgetting_mean']:>7.3f} ± {d['mean_forgetting_std']:.3f}    "
              f"{d['mean_target_conf_mean']:>7.4f} ± {d['mean_target_conf_std']:.4f}")

    print("\nPer-seed breakdown:")
    for name in rule_names:
        print(f"  {name}:")
        d = agg[name]
        for metric_pretty, metric_key in [('margin', 'mean_margin_per_seed'),
                                          ('forget', 'mean_forgetting_per_seed'),
                                          ('conf',   'mean_target_conf_per_seed')]:
            vals = d[metric_key]
            print(f"    {metric_pretty}: {[f'{v:.4f}' for v in vals]}")

    # LaTeX table
    print("\n--- LaTeX (paste into Table tab:diagnostic) ---")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Selection rule & Mean margin $\downarrow$ & Forgetting events $\uparrow$ & Mean target conf. \\")
    print(r"\midrule")
    for name in rule_names:
        d = agg[name]
        print(f"{name} & "
              f"${d['mean_margin_mean']:.3f} \\pm {d['mean_margin_std']:.3f}$ & "
              f"${d['mean_forgetting_mean']:.2f} \\pm {d['mean_forgetting_std']:.2f}$ & "
              f"${d['mean_target_conf_mean']:.3f} \\pm {d['mean_target_conf_std']:.3f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag_dir', type=str, required=True,
                        help='Directory containing cgr_diag_seed*.pt files.')
    parser.add_argument('--E', type=int, default=4,
                        help='CGR variance window (should match what was used in training).')
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help='Buffer size used in the run; controls per-class top-K.')
    parser.add_argument('--last_k_for_margin', type=int, default=5,
                        help='Average margin over the last K training epochs for the report column.')
    args = parser.parse_args()

    logs = load_seed_logs(args.diag_dir)
    print(f"Loaded {len(logs)} seed logs from {args.diag_dir}")
    for log in logs:
        print(f"  seed={log['seed']}  E={log['E']}  n_epochs={log['n_epochs']}  "
              f"n_samples={log['n_sample_per_task']}  buffer_size={log['buffer_size']}")

    # (b.1) cross-seed
    if len(logs) >= 2:
        mean_rho, std_rho, all_rhos, pairs = cross_seed_spearman(logs, args.E)
        print_b1(mean_rho, std_rho, all_rhos, pairs, len(logs))
    else:
        print("\n(b.1) Cross-seed correlation skipped: need >= 2 seeds.")

    # (a.2) variance vs forgetting -- across all seeds
    a2_results, a2_mean, a2_std = variance_vs_forgetting_per_seed(logs, args.E)
    print_a2(a2_results, a2_mean, a2_std)

    # (b.2) diagnostic table -- averaged across all seeds
    agg, per_seed_rows, k_per_class, num_classes = diagnostic_table_all_seeds(
        logs, args.E, args.buffer_size, last_k_for_margin=args.last_k_for_margin
    )
    print_b2(agg, per_seed_rows, k_per_class, num_classes, len(logs))


if __name__ == '__main__':
    main()
