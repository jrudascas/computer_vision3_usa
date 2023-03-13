import PIL
import os
import os.path
from PIL import Image

img_dir = r'./TotalFaces/'

print('Bulk images resizing started...')
const = 0

for img in os.listdir(img_dir):
	f_img = img_dir + img
	splitName = img.split('.')
	f, e = os.path.splitext(img_dir + img)
	img = Image.open(f_img)
	img = img.convert('RGB')
	img = img.resize((50, 50))
	img.save('./Resize/' + splitName[0] + str(const) + '_resized.jpeg')
	const += 1