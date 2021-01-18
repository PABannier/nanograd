# TO MAKE IT EXECUTABLE: chmod -x run_tests.sh
python3 -m pytest ./tests/test_ops_cpu.py
python3 -m pytest ./tests/test_ops_gpu.py
python3 -m pytest ./tests/test_loss.py
python3 -m pytest ./tests/test_mlp_cpu.py
python3 -m pytest ./tests/test_conv.py