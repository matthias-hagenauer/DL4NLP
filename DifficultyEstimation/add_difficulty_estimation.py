import json
import argparse
from sentinel_metric import download_model, load_from_checkpoint

model_path = download_model("Prosho/sentinel-src-24")
model = load_from_checkpoint(model_path)


REQUIRED_FIELDS = {"lp", "source", "target"}
def compute_difficulty_scores(sources):
            # turn the sources into the right format for the model
            inputs = [{"src": src} for src in sources]
            # get the scores from the model
            scores = model.predict(inputs, batch_size=8, gpus=1).scores

            return scores


def add_difficulty_scores(args):
    # Load the list of json objects from the file
    json_lines = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                json_lines.append(rec)
    
    # Extract the source texts
    sources = [rec["source"] for rec in json_lines]

    # Compute difficulty scores
    scores = compute_difficulty_scores(sources)
    
    # Add scores to the json objects
    for rec, score in zip(json_lines, scores):
        rec["difficulty_score"] = score
    
    # Write the list of json objects back to a jsonl file
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in json_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(args):
    add_difficulty_scores(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file")
    args = parser.parse_args()

    main(args)
