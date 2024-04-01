# README

<!-- > [!NOTE]
> The current version is still in testing. A tested version is in the [raptors](https://github.com/ZIYU-DEEP/bilevel-reasoner/tree/raptors) branch, where we have the benchmark results with Llemma-7b on `bfs_low` and `bfs_low_with_raw_high` methods. The latter is very much underperforming (23 out of 244 problems), likely due to the fact that the raw informal proof is unstructured and the model capacity is bad. A proof of concept colab with GPT-4 can be found at [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BI3u6NwVtefTpWzQNj-OwPf6C3ONuPFn?usp=sharing). -->

## Setup

```bash
conda create --name rir python=3.10
conda activate rir
pip install -r requirements.txt  # You may ignore the vllm installation if you do not have cuda
```

Optional installation guide for SMT is in [`INSTALLATION.md`](https://github.com/ZIYU-DEEP/bilevel-reasoner/blob/main/INSTALLATION.md).

## Run

### Running with OpenAI API

Make sure you have set up the API key in your environment before running.

```bash
echo 'export OPENAI_API_KEY="your_api_key"' >> ~/.bashrc
source ~/.bashrc
```

```bash
# Low-level search
python run.py --config_name dojo_test.yaml --search_method bfs_low --gen_method openai

# Bi-level search
python run.py --config_name dojo_test.yaml --search_method bfs_bilevel --gen_method openai
```

### Running with Open-Source Models

```bash
# Low-level search
python run.py --config_name dojo_test.yaml --search_method bfs_low

# Bi-level search
python run.py --config_name dojo_test.yaml --search_method bfs_bilevel
```

## To Reproduce the Issue on MiniF2F

### Command

```bash
python run.py --config_name dojo_test.yaml --search_method bfs_low --gen_method openai --slice_size 1 -mn gpt-4-0125-preview
```

This command will apply `dojo_test.yaml` as the configuration file, which uses `./data/minif2f_lean4_dojo.jsonl` to trace the [miniF2F-lean4 repo](https://github.com/yangky11/miniF2F-lean4/tree/d4ec261d2b9b8844f4ebfad4253cf3f42519c098).

### Error Messages

```bash
... (omitted)
[1513/1514] Building MiniF2F
2024-03-27 21:41:35.736 | INFO     | __main__:main:188 - Tracing miniF2F-lean4
2024-03-27 21:41:35.851 | DEBUG    | __main__:main:193 - lake env lean --threads 32 --run ExtractData.lean
 99%|████████████████████████████████████████████████████████████████████████████████████ | 2349/2374 [09:50<01:15,  3.04s/it]2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Macro.ast.json
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Conv.ast.json
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Euclidean.ast.json
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Component/Panel/GoalTypePanel.dep_paths
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Conv.dep_paths
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/ExprPresentation.ast.json
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Component/Panel/SelectionPanel.ast.json
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Venn.dep_paths
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Component/Panel/SelectionPanel.dep_paths
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/InteractiveSvg.ast.json
2024-03-27 21:51:35.814 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/RbTree.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Jsx.ast.json
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/SelectInsertConv.ast.json
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Euclidean.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Dynkin.ast.json
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Component/Panel/GoalTypePanel.ast.json
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Component/InteractiveSvg.ast.json
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Dynkin.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Rubiks.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/InteractiveSvg.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Venn.ast.json
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/ExprPresentation.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/SelectInsertConv.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Macro.dep_paths
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Svg.ast.json
2024-03-27 21:51:35.815 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets.ast.json
2024-03-27 21:51:35.816 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Component/InteractiveSvg.dep_paths
2024-03-27 21:51:35.816 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/RbTree.ast.json
2024-03-27 21:51:35.816 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Plot.ast.json
2024-03-27 21:51:35.816 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Plot.dep_paths
2024-03-27 21:51:35.816 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Svg.dep_paths
2024-03-27 21:51:35.816 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Jsx.dep_paths
2024-03-27 21:51:35.816 | WARNING  | __main__:check_files:132 - Missing /tmp/tmpdacwmlc2/workspace/miniF2F-lean4/.lake/packages/proofwidgets/.lake/build/ir/ProofWidgets/Demos/Rubiks.ast.json
2024-03-27 21:51:36,696	INFO worker.py:1540 -- Connecting to existing Ray cluster at address: 130.207.126.85:6379...
  0%|                                                                                                 | 0/244 [25:46<?, ?it/s]
Traceback (most recent call last):
  File "/localscratch/cat/github/bilevel-reasoner/run.py", line 249, in <module>
    main()
  File "/localscratch/cat/github/bilevel-reasoner/run.py", line 193, in main
    attempt_results = search.proof_search(theorem=theorem,
  File "/localscratch/cat/github/bilevel-reasoner/utils/search.py", line 95, in proof_search
    with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/lean_dojo/interaction/dojo.py", line 265, in __enter__
    raise ex
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/lean_dojo/interaction/dojo.py", line 185, in __enter__
    traced_repo_path = get_traced_repo_path(self.repo)
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/lean_dojo/data_extraction/trace.py", line 83, in get_traced_repo_path
    traced_repo = TracedRepo.from_traced_files(tmp_dir / repo.name, build_deps)
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/lean_dojo/data_extraction/traced_data.py", line 1098, in from_traced_files
    with ray_actor_pool(_TracedRepoHelper, root_dir, repo) as pool:
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/contextlib.py", line 135, in __enter__
    return next(self.gen)
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/lean_dojo/utils.py", line 73, in ray_actor_pool
    ray.init()
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/ray/_private/worker.py", line 1680, in init
    _global_node = ray._private.node.Node(
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/ray/_private/node.py", line 198, in __init__
    node_ip_address = self._wait_and_get_for_node_address()
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/ray/_private/node.py", line 992, in _wait_and_get_for_node_address
    node_ip_address = ray._private.services.get_cached_node_ip_address(
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/ray/_private/services.py", line 693, in get_cached_node_ip_address
    with FileLock(str(file_path.absolute()) + ".lock"):
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/filelock/_api.py", line 297, in __enter__
    self.acquire()
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/filelock/_api.py", line 255, in acquire
    self._acquire()
  File "/nethome/cat/anaconda3/envs/rir/lib/python3.10/site-packages/filelock/_unix.py", line 39, in _acquire
    fd = os.open(self.lock_file, open_flags, self._context.mode)
PermissionError: [Errno 13] Permission denied: '/tmp/ray/session_2024-03-26_22-54-31_191753_1492401/node_ip_address.json.lock'
```

It seems that the error messages contain two parts:

1. Missing files of `.ast.json` and `.dep_paths`. (I also tried directly using `lake build` under the cloned [miniF2F-lean4 repo](https://github.com/yangky11/miniF2F-lean4/tree/d4ec261d2b9b8844f4ebfad4253cf3f42519c098), and it seems that those files are not contained in the build directory).
2. The permission error for the `/tmp/ray/.../node_ip_address.json.lock` file.

(Using `lean_dojo==1.1.2` with the [lean-dojo-mew](https://github.com/rah4927/lean-dojo-mew) repo is fine, though. The experiments can be reproduced with the command in the main branch.)
