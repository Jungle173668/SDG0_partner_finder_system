"""
Champion-Challenger model selection.

Queries all MLflow evaluation runs, finds the method with the highest F1,
registers it as a new version of 'sdg-classifier', and marks it Production.
The previous Production version is automatically archived.

Usage
-----
    python -m ml.select_champion           # dry run — print winner, don't register
    python -m ml.select_champion --promote # register and promote to Production
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def select_champion(promote: bool = False) -> dict:
    import dagshub
    import mlflow

    dagshub.init(
        repo_owner="jungle173770",
        repo_name="SDG0_partner_finder_system",
        mlflow=True,
    )

    # -------------------------------------------------------------------------
    # Find the best run across all evaluate_* runs
    # -------------------------------------------------------------------------
    runs = mlflow.search_runs(
        filter_string="tags.mlflow.runName LIKE 'eval_%'",
        order_by=["metrics.f1 DESC"],
    )

    if runs.empty:
        print("No evaluation runs found. Run: python -m ml.sdg_classifier evaluate --method llm")
        return {}

    best = runs.iloc[0]
    best_run_id = best["run_id"]
    best_method  = best.get("params.method", "unknown")
    best_f1      = best.get("metrics.f1", 0.0)

    print(f"\n{'=' * 55}")
    print(f"  CHAMPION SELECTION")
    print(f"{'=' * 55}")
    print(f"  Best method : {best_method}")
    print(f"  F1          : {best_f1:.4f}")
    print(f"  Run ID      : {best_run_id[:16]}...")
    print(f"{'=' * 55}")

    # Show all methods for comparison
    print("\n  All evaluated methods:")
    for _, row in runs.iterrows():
        marker = "← champion" if row["run_id"] == best_run_id else ""
        print(f"    {row.get('params.method', '?'):12s}  F1={row.get('metrics.f1', 0):.4f}  {marker}")

    if not promote:
        print("\n[DRY RUN] Pass --promote to register and mark as Production.")
        return {"method": best_method, "f1": best_f1, "run_id": best_run_id, "promoted": False}

    # -------------------------------------------------------------------------
    # Register new version and promote to Production
    # -------------------------------------------------------------------------
    client = mlflow.tracking.MlflowClient()

    # Get artifact URI for the best run
    artifact_uri = client.get_run(best_run_id).info.artifact_uri
    artifacts = [a.path for a in client.list_artifacts(best_run_id)]

    # Choose source artifact based on method
    if "prompt_v1.txt" in artifacts:
        source = f"{artifact_uri}/prompt_v1.txt"
    elif "model" in artifacts:
        source = f"{artifact_uri}/model"
    else:
        source = artifact_uri  # fallback: point to run root

    mv = client.create_model_version(
        name="sdg-classifier",
        source=source,
        run_id=best_run_id,
        description=f"{best_method}, F1={best_f1:.4f}",
    )

    # Archive all previous Production versions
    existing = client.search_model_versions("name='sdg-classifier'")
    for v in existing:
        if v.current_stage == "Production" and v.version != mv.version:
            client.transition_model_version_stage("sdg-classifier", v.version, "Archived")
            print(f"\n  Archived previous champion: v{v.version}")

    # Promote new version
    client.transition_model_version_stage("sdg-classifier", mv.version, "Production")
    print(f"  Promoted to Production: sdg-classifier v{mv.version} ({best_method}, F1={best_f1:.4f})")

    return {"method": best_method, "f1": best_f1, "run_id": best_run_id, "version": mv.version, "promoted": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and promote the best SDG classifier")
    parser.add_argument("--promote", action="store_true", help="Register and promote champion to Production")
    args = parser.parse_args()

    result = select_champion(promote=args.promote)
    if result:
        import json
        print(f"\nResult: {json.dumps(result, indent=2)}")
