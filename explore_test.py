from datasets import load_dataset
try:
    ds = load_dataset("uw-math-ai/theorem-search-dataset", "theorem-test")
    print(ds)
    print(ds['test'][0] if 'test' in ds else ds['train'][0])
except Exception as e:
    print(e)
