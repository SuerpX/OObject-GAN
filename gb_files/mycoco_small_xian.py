import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import gaussian_filter
import pdb
from random import randint, uniform
import torch


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, categories=None, batch_size=32, transition = False):
        from pycocotools.coco import COCO
        self.bsize = batch_size
        self.root = root
        self.coco = COCO(annFile)
        self.categories = categories
#         if categories is None:
        self.cat_ids = []
        self.transition = transition
        if categories is None:
            self.ids = list(self.coco.imgs.keys())
        else:
            imgIds = []
            for cat in categories:
                catIds = self.coco.getCatIds(catNms=[cat])
                ## Filter if the mask is too small 
                filterSmall = True
                if(filterSmall):
                    for imgid in self.coco.getImgIds(catIds=catIds ):
                        if(self.checkMaskSize(imgid,catIds)):
                            imgIds.extend([imgid])
                else:
                    imgIds.extend(self.coco.getImgIds(catIds=catIds ))
                    
                    
                self.cat_ids.append(catIds[0])                
            self.ids = np.unique(imgIds).tolist()
#             pdb.set_trace()
        self.cat_ids = set(self.cat_ids)
            
        self.transform = transform
        self.target_transform = target_transform
    
    def checkMaskSize(self,imgid,catIds):
        annJSONList = self.coco.loadAnns(self.coco.getAnnIds(imgIds = [imgid], catIds =catIds))
        sizes = []
        for annJSON in annJSONList:
            sizes.append(np.sum(self.coco.annToMask(annJSON)))
#         pdb.set_trace()
        return(all(np.array(sizes)>50000) and all(np.array(sizes)<70000))
            
#         if(all([np.sum(self.coco.annToMask(annJSON))>100 for annJSON in annJSONList])):
        
    
    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        
        
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        hbox = int((rmax - rmin)*0.1)
        wbox = int((cmax - cmin)*0.1)
        
        rmin = np.clip(rmin-hbox, 0, img.shape[0]-1)
        rmax = np.clip(rmax+hbox, 0, img.shape[0]-1)
        cmin = np.clip(cmin-wbox, 0, img.shape[1]-1)
        cmax = np.clip(cmax+wbox, 0, img.shape[1]-1)
        
        return rmin, rmax, cmin, cmax

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        target_ids = -1
        if len(target)>0:
            target_objs = [t for t in target if t['category_id'] in self.cat_ids]
            npmask = coco.annToMask(np.random.choice(target_objs))*230 # small trick to prevent the discriminator
            # from checking if the mask's value is 1 or not to detect fake masks
        

            mask = Image.fromarray(npmask) # because target_transform 
            # will scale down the range to [0...1], so we have to multiply by 255
#             box = self.bbox(npmask)

#             patch = npmask[box[0]:box[1], box[2]:box[3]]
#             loc_enc[box[0]:box[1], box[2]:box[3]] = gaussian_filter(patch*1.0, ((box[1]-box[0])*0.2, (box[3]-box[2])*0.2))
#             loc_enc = Image.fromarray(loc_enc)
            
        else:
            mask = Image.fromarray(np.zeros((1,1), dtype=np.uint8))
#             loc_enc = Image.fromarray(np.zeros((1,1), dtype=np.uint8))
        
        if self.transform is not None:
            img = self.transform(img)
# 
        if self.target_transform is not None:
            mask = self.target_transform(mask)
#             loc_enc = self.target_transform(loc_enc)
        
        obj =  np.multiply(img, mask)
        #print(mask.shape)
        IMAGE_SIZE = mask.shape[1]
        if self.transition:
            r = randint(0,3)
            rmin, rmax, cmin, cmax =  self.bbox(mask[0].data.cpu().numpy())
            obj_trans_part = obj[:, rmin : rmax, cmin : cmax]
            mask_trans_part = mask[:, rmin : rmax, cmin : cmax]
            if r == 0:
                transform_obj_mask = transforms.Compose([transforms.ToPILImage(),
                                                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                       transforms.ToTensor()
                                                       ])
                mask_trans = transform_obj_mask(mask_trans_part)
                obj_trans = transform_obj_mask(obj_trans_part)
            if r == 1:
                
                cx = randint(0 - cmin,  IMAGE_SIZE - cmax )
                ry = randint(0 - rmin,  IMAGE_SIZE - rmax )
                mask_trans = torch.zeros((1,IMAGE_SIZE, IMAGE_SIZE))
                obj_trans = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

                mask_trans[:, rmin + ry : rmax + ry, cmin + cx : cmax + cx] = mask_trans_part
                obj_trans[:, rmin + ry : rmax + ry, cmin + cx : cmax + cx] = obj_trans_part
                
            if r == 2:
                degree = uniform(0, 45)
                transform_obj_mask = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomRotation(degree),
                                       transforms.ToTensor()
                                       ])
                mask_trans = transform_obj_mask(mask)
                obj_trans = transform_obj_mask(obj)
            if r == 3:
                mask_trans = mask
                obj_trans = obj

            return img, mask, obj, mask_trans, obj_trans#, loc_enc
        else:
            return img, mask


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
