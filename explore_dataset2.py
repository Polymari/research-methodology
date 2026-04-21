import json
from datasets import load_dataset
try:
    ds = load_dataset("uw-math-ai/theorem-search-dataset")
    info = {"keys": list(ds.keys())}
    for k in ds.keys():
        info[k] = {"size": len(ds[k]), "features": list(ds[k].features.keys())}
    with open("dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
except Exception as e:
    with open("dataset_info.json", "w") as f:
        f.write(str(e))
