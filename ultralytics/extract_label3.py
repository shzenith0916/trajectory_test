import sys
import os
import shutil

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'c:\Users\USER\Documents\AKAS관련\AKAS1.0_YOLO연구용\ultralytics\data\all'
OUT_BASE = r'c:\Users\USER\Documents\AKAS관련\AKAS1.0_YOLO연구용\ultralytics\data\label3'


def has_label3(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            if class_id == 3:
                return True
    return False


total_copied = 0

for split in ['train', 'valid']:
    img_in = os.path.join(BASE, split, 'images')
    lbl_in = os.path.join(BASE, split, 'labels')
    img_out = os.path.join(OUT_BASE, split, 'images')
    lbl_out = os.path.join(OUT_BASE, split, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    label_files = [f for f in os.listdir(lbl_in) if f.endswith('.txt')]
    count = 0

    for lbl_file in label_files:
        lbl_path = os.path.join(lbl_in, lbl_file)
        if not has_label3(lbl_path):
            continue

        stem = os.path.splitext(lbl_file)[0]
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = stem + ext
            if os.path.exists(os.path.join(img_in, candidate)):
                img_file = candidate
                break

        if img_file is None:
            print(f"[WARNING] no image found: {stem}")
            continue

        shutil.copy2(os.path.join(img_in, img_file),
                     os.path.join(img_out, img_file))
        shutil.copy2(lbl_path, os.path.join(lbl_out, lbl_file))
        count += 1

    print(f"{split}: {count} files copied")
    total_copied += count

print(f"\nTotal: {total_copied} pairs copied")
print(f"Output: {OUT_BASE}")
