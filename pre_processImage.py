import numpy as np
from PIL import Image
import os


def convertImage():
    listpath = os.listdir(r"F:\数据集\小黄人项目\bg_pic2")
    for path in listpath:
        img = Image.open(os.path.join(r"F:\数据集\小黄人项目\bg_pic2", path))
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.ANTIALIAS)
        img.save(os.path.join(r"F:\数据集\小黄人项目\datas", path))

def createDataset(dirimage):
    listpath = os.listdir(dirimage)
    for index, path in enumerate(listpath):
        img = Image.open(os.path.join(dirimage, path))
        if index < 2000 or (index >= 4000 and index < 4500):
            minions = Image.open("F:\数据集\小黄人项目\yellow/{}.png".format(np.random.randint(1, 21)))
            # 缩放
            h = w = np.random.randint(64, 180)
            minions = minions.resize((h, w), Image.ANTIALIAS)
            # 旋转
            minions = minions.rotate(np.random.randint(-30, 30))
            # 翻转
            minions = minions.transpose(Image.FLIP_LEFT_RIGHT) if np.random.randint(0, 2) == 1 else minions
            x, y = np.random.randint(0, 224 - w), np.random.randint(0, 224 - h)
            # 掩码
            r, g, b, a = minions.split()
            img.paste(minions, (x, y), mask=a)

            if not os.path.isdir("datasets"):
                os.mkdir("datasets")
            img.save("datasets/{}.{}.{}.{}.{}.{}.jpg".format(index, x, y, x + w, y + h, 1))
        else:
            img.save("datasets/{}.0.0.0.0.0.jpg".format(index))

if __name__ == '__main__':
    # convertImage()
    createDataset(r"F:\数据集\小黄人项目\datas")