import os
import sys
import shutil

def create_imagenet_ext(src_path, tgt_path, num_class=20):
	if not os.path.exists(tgt_path):
		os.mkdir(tgt_path)
	else:
		shutil.rmtree(tgt_path)

	for i, img_dir in enumerate(os.listdir(src_path)):
		shutil.copytree(os.path.join(src_path, img_dir), os.path.join(tgt_path, img_dir))
		if i == num_class-1:
			break

src_path = '/media/cchen/StorageDisk/imagenet/raw-data/train'
tgt_path = '/media/cchen/StorageDisk/yzhao/GAN/BigGAN-PyTorch/data/Ext'
create_imagenet_ext(src_path, tgt_path)