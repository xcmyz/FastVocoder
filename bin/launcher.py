import os
import sys
import argparse
sys.path.append(".")


if __name__ == "__main__":
    from bin import run_train, run_preprocess, run_synthesizer, run_publisher, run_test
    MODE = os.getenv("MODE")
    if MODE == "train":
        run_train()
    elif MODE == "preprocess":
        run_preprocess()
    elif MODE == "synthesize":
        run_synthesizer()
    elif MODE == "publish":
        run_publisher()
    elif MODE == "test":
        run_test()
