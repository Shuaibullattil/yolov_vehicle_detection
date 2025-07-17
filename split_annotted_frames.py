import os, random, shutil

IMG_DIR = "datasets/images"
LBL_DIR = "datasets/labels"

train_img_dir = "datasets/images/train"
val_img_dir = "datasets/images/val"
train_lbl_dir = "datasets/labels/train"
val_lbl_dir = "datasets/labels/val"

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg") or f.endswith(".png")]
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

for img in train_images:
    shutil.copy(os.path.join(IMG_DIR, img), train_img_dir)
    shutil.copy(os.path.join(LBL_DIR, img.replace(".jpg", ".txt").replace(".png", ".txt")), train_lbl_dir)

for img in val_images:
    shutil.copy(os.path.join(IMG_DIR, img), val_img_dir)
    shutil.copy(os.path.join(LBL_DIR, img.replace(".jpg", ".txt").replace(".png", ".txt")), val_lbl_dir)
