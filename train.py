import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--project", type=str, default="runs/train")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.img, project=args.project)
    print("Training complete. Check runs/train for results.")

if _name_ == "_main_":
   main()