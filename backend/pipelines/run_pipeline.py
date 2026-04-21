from backend.pipelines.steps import (
    embed_jobs,
    reduce_dimension_jobs,
    cluster_jobs,
    generate_job_descriptions,
    embed_clusters
)
import backend.app.database as database
from typing import Optional
import argparse

PIPELINE_STEPS = [
    ("Embed Jobs", embed_jobs.run),
    ("Reduce Dimensions", reduce_dimension_jobs.run),
    ("Cluster Jobs", cluster_jobs.run),
    ("Generate Job Descriptions", generate_job_descriptions.run),
    ("Embed Clusters", embed_clusters.run)
]

def run_pipeline(step_name: Optional[str] = None):
    # Create new database session instance
    SessionLocal = database.SessionLocal
    db_session = SessionLocal()

    # If a specific step is provided, run only that step
    if step_name:
        step_func = dict(PIPELINE_STEPS).get(step_name)
        if not step_func:
            print(f"Step '{step_name}' not found in pipeline.")
            return
        
        # Try to run the specified step and catch any exceptions
        try:
            step_func(db_session)
        except Exception as e:
            print(f"Pipeline failed at step: {step_name}")
            print(f"Error: {e}")
        else:
            print(f"Completed: {step_name}")
        return

    # Run all steps in sequence if no specific step is provided
    for step_name, step_func in PIPELINE_STEPS:
        print(f"Running step: {step_name}")

        # Run the step and catch any exceptions to prevent the entire pipeline from crashing
        try:
            step_func(db_session)
        except Exception as e:
            print(f"Pipeline failed at step: {step_name}")
            print(f"Error: {e}")
            break
        else:
            print(f"Completed: {step_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CareerAlign pipeline")
    parser.add_argument("--step", type=str, help="Run a specific pipeline step")

    args = parser.parse_args()

    run_pipeline(step_name=args.step)