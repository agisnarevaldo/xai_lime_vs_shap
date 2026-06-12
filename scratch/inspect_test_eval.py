import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("02_finetuning_indobert_emotion2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [16, 17, 18]:
    cell = nb['cells'][idx]
    print(f"\n=== Cell {idx} ({cell['cell_type']}) ===")
    print("Source:")
    print("".join(cell.get("source", []))[:500])
    print("\nOutputs:")
    for out in cell.get("outputs", []):
        out_type = out.get("output_type")
        if out_type == "stream":
            print("".join(out.get("text", [])))
        elif out_type == "execute_result":
            print("".join(out.get("data", {}).get("text/plain", [])))
        elif out_type == "error":
            print("ERROR:", out.get("ename"), out.get("evalue"))
            print("\n".join(out.get("traceback", [])))
    print("-" * 50)
