"""期刊实验主入口；默认只在显式 train/extract 命令时运行模型。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from journal_detector.__main__ import main
if __name__=='__main__':main()
