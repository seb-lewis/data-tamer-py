import os
import sys

# Ensure project root is on sys.path so `import data_tamer` works without install
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

