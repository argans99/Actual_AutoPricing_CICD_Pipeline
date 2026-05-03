import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_info_output_path", type=str, required=True)
    args, _ = parser.parse_known_args()
    return args


def main(args):
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")

    os.makedirs(args.model_info_output_path, exist_ok=True)

    model_info = {
        "name": args.model_name,
        "path": args.model_path,
        "type": "mlflow_model",
    }

    output_path = os.path.join(args.model_info_output_path, "model_info.json")

    with open(output_path, "w") as of:
        json.dump(model_info, of)

    print(f"Model info written to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
