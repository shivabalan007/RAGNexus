from __future__ import annotations


def check_thresholds(report: dict, thresholds: dict[str, float]) -> list[str]:
    """Return human-readable failures. Thresholds are minimum acceptable averages."""
    summary = report.get("summary", {})
    failures = []
    for metric, minimum in thresholds.items():
        if metric not in summary:
            failures.append(f"Missing metric '{metric}' in report")
        elif float(summary[metric]) < float(minimum):
            failures.append(f"{metric}={summary[metric]:.4f} is below minimum {minimum:.4f}")
    return failures
