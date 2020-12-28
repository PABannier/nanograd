# TO MAKE IT EXECUTABLE: chmod -x run_tests.sh
python3 -m pytest test_ops.py
python3 -m pytest test_loss.py
python3 -m pytest test_mlp.py
python3 -m pytest test_conv.py