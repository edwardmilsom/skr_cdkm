"""
stuff to get test discovery to work, and allow import of test utils
"""
import os
import sys
PROJECT_PATH = os.getcwd()
DKM_PATH = os.path.join(PROJECT_PATH, "dkm")
TESTS_PATH = os.path.join(PROJECT_PATH, "tests")
sys.path.append(DKM_PATH); sys.path.append(TESTS_PATH)