import sys
from pathlib import Path

current_file = Path(__file__).resolve()
testfiles_path = current_file.parent.parent / 'src'
sys.path.insert(0, str(testfiles_path))