from __future__ import annotations

import os
import pandas as pd


def export_results(
    round_details: list,
    experiment_summaries: list,
    output_dir: str = "results",
    per_class_records: list | None = None,
) -> str:
    """Export experiment results to an Excel file with up to three sheets."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "experiment_results.xlsx")

    df_details = pd.DataFrame(round_details)
    df_summary = pd.DataFrame(experiment_summaries)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_details.to_excel(
            writer, sheet_name="round_shapley_details", index=False
        )
        df_summary.to_excel(
            writer, sheet_name="experiment_summary", index=False
        )
        if per_class_records:
            df_pc = pd.DataFrame(per_class_records)
            df_pc.to_excel(
                writer, sheet_name="per_class_records", index=False
            )

    return filepath
