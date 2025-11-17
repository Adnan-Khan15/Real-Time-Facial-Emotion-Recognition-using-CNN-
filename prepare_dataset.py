import cv2
import os
from tqdm import tqdm

def prepare_folder(input_dir, output_dir, size=96):
    os.makedirs(output_dir, exist_ok=True)

    for cls in os.listdir(input_dir):
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)

        os.makedirs(cls_output, exist_ok=True)

        for img_name in tqdm(os.listdir(cls_input), desc=f"Processing {cls}"):
            img_path = os.path.join(cls_input, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.equalizeHist(img)

            # Resize to 96x96
            img = cv2.resize(img, (size, size))

            # Convert grayscale → RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Save
            cv2.imwrite(os.path.join(cls_output, img_name), img)


prepare_folder("data/train", "data_processed/train")
prepare_folder("data/val", "data_processed/val")
prepare_folder("data/test", "data_processed/test")

print("Dataset preprocessing complete!")
