import random
import os, glob
from straug.process import Posterize, AutoContrast, Sharpness
from straug.blur import MotionBlur
from straug.camera import Pixelate, Contrast, Brightness, JpegCompression
from straug.weather import Shadow, Fog
from PIL import Image

def augument(image):
    if random.random() < 0.1:
        if random.random() > 0.5:
            image = Fog()(image, mag=random.choice(range(-10,10)))
        return MotionBlur()(image, mag=random.choice(range(-10,10)))
    elif 0.1 <= random.random() < 0.2:
        if random.random() > 0.5:
            image = Brightness()(image, mag=random.choice(range(-10,10)))
        return Fog()(image, mag=random.choice(range(-10,10)))
    elif 0.2 <= random.random() < 0.3:
        if random.random() > 0.5:
            image = Fog()(image, mag=random.choice(range(-10,10)))
        return Posterize()(image, mag=random.choice(range(-10,10)))
    elif 0.3 <= random.random() < 0.4:
        if random.random() > 0.5:
            image = Brightness()(image, mag=random.choice(range(-10,10)))
        return AutoContrast()(image, mag=random.choice(range(-10,10)))
    elif 0.4 <= random.random() < 0.5:
        if random.random() > 0.5:
            image = Brightness()(image, mag=random.choice(range(-10,10)))
        return Shadow()(image, mag=random.choice(range(-10,10)))
    elif 0.5 <= random.random() < 0.6:
        if random.random() > 0.5:
            image = Shadow()(image, mag=random.choice(range(-10,10)))
        return Pixelate()(image, mag=random.choice(range(-10,10)))
    elif 0.6 <= random.random() < 0.7:
        if random.random() > 0.5:
            image = Pixelate()(image, mag=random.choice(range(-10,10)))
        return Contrast()(image, mag=random.choice(range(-10,10)))
    elif 0.7 <= random.random() < 0.8:
        if random.random() > 0.5:
            image = Shadow()(image, mag=random.choice(range(-10,10)))
        return Brightness()(image, mag=random.choice(range(-10,10)))
    elif 0.8 <= random.random() < 0.9:
        if random.random() > 0.5:
            image = Sharpness()(image, mag=random.choice(range(-10,10)))
        return JpegCompression()(image, mag=random.choice(range(-10,10)))
    else:
        if random.random() > 0.5:
            image = Brightness()(image, mag=random.choice(range(-10,10)))
        return Sharpness()(image, mag=random.choice(range(-10,10)))


def aug_image(img_path, logs_path):
    save_path = img_path.replace("/home/viethq/viet/OCR/training_data/images", logs_path)
    image_image = os.path.basename(image_path)
    os.makedirs(save_path.replace(image_image, ""), exist_ok=True)
    with open(img_path, 'rb') as img_file:
        img = Image.open(img_file).convert('RGB')
        if random.random() > 0.8:
            w, h = img.size
            img = img.crop((random.randint(0, 10), random.randint(0, 10), random.randint(int(w-15), int(w)), random.randint(int(h-10), int(h))))
        
        img_aug = augument(img)
        img_aug.save(save_path)
    return save_path

if __name__ == "__main__":
    data_path = "/home/viethq/viet/OCR"
    logs_path = f"{data_path}/kalapa_aug"
    train_data_path = os.path.join(data_path, "training_data")
    annotations = glob.glob(os.path.join(train_data_path, "annotations", "*.txt"))
    image_dir = os.path.join(train_data_path, "images")
    os.makedirs(logs_path, exist_ok=True)
    f = open(os.path.join(logs_path, "labels.txt"), "w")
    for i in range(10):
        for anno_path in annotations[:2]:
            lines = open(anno_path, "r").readlines()
            for line in lines[:1]:
                relative_path, label = line.strip().split(maxsplit=1)
                image_path = os.path.join(image_dir, relative_path)
                save_path = aug_image(image_path, f"{data_path}/kalapa_aug/images_{i}")

                f.write("{}\t{}\n".format(save_path, label))
