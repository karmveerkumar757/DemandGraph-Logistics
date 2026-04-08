from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logistics_optimization.db.database import init_db


if __name__ == "__main__":
    init_db()
    print("SQLite database initialized.")
