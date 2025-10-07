import sys
from mini_model import MiniModel
from chap_core.model_spec import get_dataclass
from chap_core.datatypes import create_tsdataclass

def main():
    dc = get_dataclass(estimator)
    print("-----------___________inside main")
    argv = sys.argv
    print("-----", argv)
    print("type: ", type(argv[2]))
    if not argv[1:]:
        raise SystemExit("usage: train|predict ...")
    
    model = MiniModel()
    cmd, *rest = argv 
    if cmd == "train":
        model.train(*rest)
    elif cmd == "predict":
        model.predict(*rest)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()