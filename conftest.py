# conftest.py  ← à créer à la racine /home/abakar/TP_MLOps/
import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))
