"""
Setup condiviso: rende importabili common.py e ranker.py (vivono solo dentro
ocx_relevance/, le altre pipeline li raggiungono con lo stesso sys.path.insert —
vedi corpus/fetch_corpus.py, ai_relevance/06_rank.py, ecc. per il pattern reale).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
