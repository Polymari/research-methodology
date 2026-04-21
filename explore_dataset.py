from datasets import load_dataset
try:
    ds = load_dataset("uw-math-ai/theorem-search-dataset")
    print("Dataset keys:", ds.keys())
    for split in ds.keys():
        print(f"Split {split} size: {len(ds[split])}")
        print("Features:", ds[split].features)
except Exception as e:
    print("Error:", e)
