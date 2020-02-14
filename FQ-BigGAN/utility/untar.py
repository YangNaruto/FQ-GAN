import tarfile

src_path = '/media/cchen/StorageDisk/yzhao/datasets/images/ImageNet/'
for fname in src_path:
	if (fname.endswith("tar.gz")):
		tar = tarfile.open(fname, "r:gz")
		tar.extractall()
		tar.close()
	elif (fname.endswith("tar")):
		tar = tarfile.open(fname, "r:")
		tar.extractall()
		tar.close()