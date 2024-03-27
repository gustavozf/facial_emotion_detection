import albumentations as abm
import cv2

HIGH_P = 0.4
MED_P = 0.2
SMALL_P = 0.1

class WeakImgSegAugmenter:
    def __init__(self):
        self.transforms = abm.Compose([
            abm.HorizontalFlip(p=HIGH_P),
            abm.RandomBrightnessContrast(p=MED_P),
            abm.Downscale(
                scale_min=0.6, scale_max=0.8,
                interpolation=cv2.INTER_AREA,
                p=SMALL_P)
        ])
    
    def __call__(self, img):
        return self.transforms(image=img)['image']
    
class StrongImgSegAugmenter(WeakImgSegAugmenter):
    def __init__(self, img_size: int = 48):
        self.transforms = abm.Compose([
            abm.HorizontalFlip(p=HIGH_P),
            abm.RandomBrightnessContrast(p=MED_P),
            abm.Downscale(
                scale_min=0.6, scale_max=0.8,
                interpolation=cv2.INTER_AREA,
                p=SMALL_P),
            abm.Rotate((-30, 30), border_mode=cv2.BORDER_CONSTANT, p=HIGH_P),
            abm.Affine(p=MED_P),
            abm.ColorJitter(p=MED_P),
            abm.RandomSizedCrop(
                min_max_height=(int(img_size * 0.64), int(img_size * 0.8)),
                height=img_size, width=img_size,
                p=SMALL_P)
        ])

class DataAugmenter:
    def __init__(self, aug_type: str = 'weak', img_size: int = 48):
        if aug_type not in {'weak', 'strong'}:
            raise ValueError(
                f'Unsupported data augmentation type: {aug_type}.'
                ' Supported options: "weak" or "strong".'
            )

        self.augmenter = self.__create_augmenter(aug_type, img_size)
    
    def __create_augmenter(self, aug_type: str, img_size: int):
        if aug_type == 'weak':
            return WeakImgSegAugmenter()
    
        return StrongImgSegAugmenter(img_size=img_size)

    def __call__(self, img):
        return self.augmenter(img)