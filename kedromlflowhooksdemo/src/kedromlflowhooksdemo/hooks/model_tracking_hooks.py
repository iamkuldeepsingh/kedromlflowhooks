from typing import Any, Dict

import mlflow
import mlflow.sklearn
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node


class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together.
    """

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Hook implementation to start an MLflow run
        with the same run_id as the Kedro pipeline run.
        """
        mlflow.start_run(run_name=run_params["run_id"])
        mlflow.log_params(run_params)

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> None:
        """Hook implementation to add model tracking after some node runs.
        In this example, we will:
        * Log the parameters after the data splitting node runs.
        * Log the model after the model training node runs.
        * Log the model's metrics after the model evaluating node runs.
        """
        if node._func_name == "split_data":
            mlflow.log_params(
                {"split_data_ratio": inputs["params:example_test_data_ratio"]}
            )

        elif node._func_name == "train_model":
            model = outputs["example_model"]
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_params(inputs["parameters"])

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        mlflow.end_run()
