import shutil
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_DIR = DATA_DIR / "expected"


@pytest.fixture
def workdir(tmp_path):
    """Temporary working directory with all test input files pre-copied."""
    for name in ("ref.tif", "ref_adjusted2target.tif", "target.tif"):
        shutil.copy(str(DATA_DIR / name), str(tmp_path / name))
    return tmp_path
