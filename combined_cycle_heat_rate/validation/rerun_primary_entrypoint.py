'Primary subset of the paper-scoped portable replay.'
import runpy,sys
from pathlib import Path
if '--primary-only' not in sys.argv:sys.argv.append('--primary-only')
runpy.run_path(str(Path(__file__).with_name('rerun_submission_results.py')),run_name='__main__')
