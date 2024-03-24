CUDA_VISIBLE_DEVICES=1 python run.py --config_name dojo_test.yaml --search_method bfs_low

CUDA_VISIBLE_DEVICES=1 python run.py --config_name dojo_test.yaml --search_method bfs_low_with_raw_high

CUDA_VISIBLE_DEVICES=1 python run.py --config_name dojo_test_mew_openai.yaml --search_method bfs_low 

python run.py --config_name dojo_test_mew_openai.yaml --search_method bfs_low --slice_size 1