import json
from pathlib import Path

path = Path(__file__).parent.absolute()


file_path = Path(path, "evaluation_set.jsonl")
with open(file_path, "r") as file:
    loaded_file = [json.loads(line) for line in file]

print(loaded_file[0].keys())
print(loaded_file[0]["meta"])
