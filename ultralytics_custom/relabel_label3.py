import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

OUT_BASE = r'c:\Users\USER\Documents\AKAS관련\AKAS1.0_YOLO연구용\ultralytics\data\FOOD_DETECT'

total = 0

for split in ['train', 'valid']:
    lbl_dir = os.path.join(OUT_BASE, split, 'labels')
    label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    count = 0

    for lbl_file in label_files:
        lbl_path = os.path.join(lbl_dir, lbl_file)
        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if int(parts[0]) == 3:
                parts[0] = '0'
                new_lines.append(' '.join(parts) + '\n')
            # class 0,1,2 는 제거 (아무것도 안 함)

        with open(lbl_path, 'w') as f:
            f.writelines(new_lines)

        count += 1

    print(f"{split}: {count} files updated")
    total += count

print(f"\nTotal: {total} files updated")
