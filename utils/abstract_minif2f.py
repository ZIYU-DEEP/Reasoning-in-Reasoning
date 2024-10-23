from lean_dojo import *
import lean_dojo
import json
from pathlib import Path
import shutil

# Move the "./minif2f_dataset" folder to "./data/minif2f"
src_dir = Path("./minif2f_dataset")
dst_dir = Path("./data/minif2f")
shutil.move(str(src_dir), str(dst_dir))

# Set the destination directory path
DST_DIR = Path("./data/minif2f")

# Load and process the train.json file
train_path = DST_DIR / "random/train.json"
proofs_train = json.load(train_path.open())
print(len(proofs_train))
print(type(proofs_train))
collect_minif2f_valid = []
collect_minif2f_test = []
for proof in proofs_train[::-1]:
    if proof["traced_tactics"]:
        if 'MiniF2F/Valid' in proof['file_path']:
            collect_minif2f_valid.append(proof)
        if 'MiniF2F/Test' in proof['file_path']:
            collect_minif2f_test.append(proof)
        if proof['full_name'] == 'mathd_algebra_478':
            print('!!!')
            print(proof)

print(len(collect_minif2f_valid))
print(len(collect_minif2f_test))

# Load and process the val.json file
val_path = DST_DIR / "random/val.json"
proofs_val = json.load(val_path.open())
for proof in proofs_val[::-1]:
    if proof["traced_tactics"]:
        if 'MiniF2F/Valid' in proof['file_path']:
            collect_minif2f_valid.append(proof)
        if 'MiniF2F/Test' in proof['file_path']:
            collect_minif2f_test.append(proof)
    if proof['full_name'] == 'mathd_algebra_478':
        print('!!!')
        print(proof)
print(len(collect_minif2f_valid))
print(len(collect_minif2f_test))

# Load and process the test.json file
test_path = DST_DIR / "random/test.json"
proofs_test = json.load(test_path.open())
for proof in proofs_test[::-1]:
    if proof["traced_tactics"]:
        if 'MiniF2F/Valid' in proof['file_path']:
            collect_minif2f_valid.append(proof)
        if 'MiniF2F/Test' in proof['file_path']:
            collect_minif2f_test.append(proof)
    if proof['full_name'] == 'mathd_algebra_478':
        print('!!!')
        print(proof)

print(len(collect_minif2f_valid))
print(len(collect_minif2f_test))

print(collect_minif2f_valid[0])

# Create the "./data/minif2f/neurips" directory if it doesn't exist
neurips_dir = DST_DIR / "neurips"
neurips_dir.mkdir(parents=True, exist_ok=True)

# Save the collected valid proofs to val.json file
with open(neurips_dir / 'val.json', 'w') as f:
    json.dump(collect_minif2f_valid, f)

# Save the collected test proofs to test.json file
with open(neurips_dir / 'test.json', 'w') as f:
    json.dump(collect_minif2f_test, f)