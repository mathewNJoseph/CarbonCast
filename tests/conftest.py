import os
import sys

# The modules in src/ import each other by bare name (import common), so put src/ on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
