"""Thread limits must be set before torch or ctranslate2 import.

Both read OMP_NUM_THREADS at load time and size their pools from it. Setting
these after the import has no effect, which is why it happens here - this
module runs before any submodule.
"""
import os

from pipeline.runtime import cpu_limit  # noqa: E402

_threads = str(cpu_limit())
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ.setdefault(_var, _threads if _var != "TOKENIZERS_PARALLELISM" else "false")

print(f"[runtime] cpu_limit={_threads} (os.cpu_count reports {os.cpu_count()})",
      flush=True)
