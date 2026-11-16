import argparse
import platform
import sys
from pathlib import Path

import torch
from ultralytics import RTDETR
from ultralytics.models.utils import ops as dops

def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='Шлях до файлу data.yaml (YOLO формат)')
    p.add_argument('--source', default=None, help='Шлях до папки з тестовими зображеннями для інференсу')
    p.add_argument('--project', default='runs', help='Папка для збереження результатів тренування та інференсу')
    p.add_argument('--name', default='rtdetr_giou4', help='Ім\'я експерименту')
    p.add_argument('--epochs', type=int, default=30, help='Кількість епох')
    p.add_argument('--batch', type=int, default=8, help='Розмір батчу')
    p.add_argument('--imgsz', type=int, default=640, help='Розмір зображення (imgsz)')
    p.add_argument('--device', default='0', help='Індекс CUDA (наприклад, 0) або "cpu"')
    p.add_argument('--weights', default='rtdetr-l.pt', help='Початкові ваги (для тренування) або готові ваги (для інференсу)')
    p.add_argument('--predict-only', action='store_true', help='Пропустити тренування і запустити лише інференс')
    return p.parse_args()


def patch_matcher(**cost):
    orig = dops.HungarianMatcher.__init__
    gains = {'class': 2, 'bbox': 5, 'giou': 2}
    gains.update(cost)

    def init(self, cost_gain=None, *a, **k):
        orig(self, cost_gain=gains, *a, **k)

    dops.HungarianMatcher.__init__ = init
    print('[patch] matcher cost gains:', gains)
    return lambda: setattr(dops.HungarianMatcher, '__init__', orig)


def train(args):
    restore = patch_matcher(giou=4)
    try:
        model = RTDETR(args.weights)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            device=args.device,
            plots=True,
            workers=4
        )
        return Path(args.project) / args.name / 'weights' / 'best.pt'
    finally:
        restore()


def predict(weights, args):
    model = RTDETR(weights)
    print(f"Запуск інференсу на {args.source}...")
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        project=args.project,
        name=f"{args.name}_predict",
        save=True,          
        save_txt=True,     
        device=args.device,
        conf=0.001,
        iou=0.6,
        max_det=300
    )


def main():
    args = get_args()
    print(f'Python {platform.python_version()} | Torch {torch.__version__} | CUDA {torch.cuda.is_available()}')
    
    if str(args.device).lower() != 'cpu' and not torch.cuda.is_available():
        sys.exit('Помилка: CUDA GPU не знайдено. Використайте пристрій з GPU або вкажіть --device cpu.')

    if not args.predict_only:
        best_weights = train(args)
    else:
        best_weights = args.weights
        print(f"Тренування пропущено. Використовуються ваги: {best_weights}")

    if args.source:
        predict(best_weights, args)


if __name__ == '__main__':
    main()