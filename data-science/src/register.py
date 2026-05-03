# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_info_output_path", type=str, required=True)

    args, _ = parser.parse_known_args()
    print(f"Arguments: {args}")
    return args


def main(args):
    """Register the trained MLflow model in Azure ML."""

    print(f"Registering model: {args.model_name}")
    print(f"Model path: {args.model_path}")

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )

    registered_model = ml_client.models.create_or_update(
        Model(
            path=args.model_path,
            name=args.model_name,
            type="mlflow_model",
            description="Used cars price prediction model trained from Azure ML pipeline",
        )
    )

    model_version = registered_model.version
    print(f"Registered model: {args.model_name}")
    print(f"Registered model version: {model_version}")

    os.makedirs(args.model_info_output_path, exist_ok=True)

    model_info = {
        "id": f"{args.model_name}:{model_version}",
        "name": args.model_name,
        "version": model_version,
        "path": args.model_path,
    }

    output_path = os.path.join(args.model_info_output_path, "model_info.json")

    with open(output_path, "w") as of:
        json.dump(model_info, of)

    print(f"Model info written to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
