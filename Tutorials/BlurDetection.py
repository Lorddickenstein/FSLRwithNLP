import cv2
import os

keyframes_path = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\keyframes'
trash_path = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\\trash'

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

for file in os.listdir(keyframes_path):
	img_path = os.path.join(keyframes_path, file)
	img = cv2.imread(img_path)
	fm = variance_of_laplacian(img)

	if fm < 1200:
		text = 'Blurry'
	else:
		text = 'Not Blurry'

	variance = "%.2f" % fm
	file_name = file.split('.')
	img_name = file_name[0] + "_" + text + str(variance) + ".jpg"
	path = os.path.join(trash_path, img_name)
	cv2.imwrite(path, img)