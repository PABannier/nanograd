# TO MAKE IT EXECUTABLE: chmod -x run_tests.sh
python3 -m pytest ./tests/test_ops.py
python3 -m pytest ./tests/test_loss.py
python3 -m pytest ./tests/test_mlp.py
python3 -m pytest ./tests/test_conv.py