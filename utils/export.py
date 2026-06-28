from __future__ import annotations

import os
import pandas as pd


def export_results(
    round_details: list,
    experiment_summaries: list,
    output_dir: str = "results",
    per_class_records: list | None = None,
    experiment_config: dict | None = None,
) -> str:
    """Export results and the exact executable configuration to Excel."""
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
        if experiment_config:
            df_config = pd.DataFrame(
                [{"parameter": key, "value": value}
                 for key, value in experiment_config.items()]
            )
            df_config.to_excel(
                writer, sheet_name="experiment_config", index=False
            )

    return filepath
