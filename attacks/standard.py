from PIL import Image
import numpy as np


class Standard(object):
    def __init__(self, img_size, num, mode=0, target=0, trigger='1'):
        super(Standard, self).__init__()
        self.img_size = img_size
        self.num = num
        self.mode = mode
        self.target = target
        self.triggerSize = int(trigger) # modify standard trigger size
        self.ratio = 0.15

    def __call__(self, img, target, backdoor, idx):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR) 
        if (self.mode == 0 and idx > self.num) or self.mode == 2:
            target, backdoor = self.target, 1
            img_np=np.array(img)
            img_np[:self.triggerSize,:self.triggerSize]=255 
            img = Image.fromarray(img_np.astype('uint8')).convert('RGB')
        return img, target, backdoor

    def set_mode(self, mode):
        self.mode = mode


