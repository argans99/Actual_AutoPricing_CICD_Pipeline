import argparse
import os
import json

import mlflow
import mlflow.sklearn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_info_output_path", type=str, required=True)

    args, _ = parser.parse_known_args()
    return args


def main(args):
    print(f"Registering model: {args.model_name}")

    # Load model
    model = mlflow.sklearn.load_model(args.model_path)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=args.model_name,
    )

    # Register model in Azure ML (this is the key line)
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{args.model_name}"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name
    )

    model_version = registered_model.version

    print(f"Registered model version: {model_version}")

    os.makedirs(args.model_info_output_path, exist_ok=True)

    model_info = {
        "id": f"{args.model_name}:{model_version}"
    }

    output_path = os.path.join(args.model_info_output_path, "model_info.json")

    with open(output_path, "w") as of:
        json.dump(model_info, of)


if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
