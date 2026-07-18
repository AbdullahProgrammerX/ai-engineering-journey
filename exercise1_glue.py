from datasets import load_dataset

dataset = load_dataset("glue", "mrpc")
print(dataset)
for i in range(5):
    print(dataset["train"][i])
