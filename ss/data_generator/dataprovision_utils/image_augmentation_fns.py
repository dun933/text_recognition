import cv2, numpy as np , random , math
random.seed()

def apply_salt_and_pepper(inputmat, amount=0.01, sp_ratio=0.5):
	print('image_augmentation_fns.apply_salt_and_pepper.amount=',amount)
	s_vs_p = sp_ratio
	# Salt mode
	img_h, img_w, _ = inputmat.shape
	size = img_h * img_w
	num_salt = math.ceil(amount * size * s_vs_p)

	for _ in range(num_salt):
		random_y = random.randint(0, img_h-1)
		random_x = random.randint(0, img_w-1)
		inputmat[random_y, random_x,:] = [255,255,255]

	num_pepper = math.ceil(amount * size * (1-s_vs_p))

	for _ in range(num_pepper):
		random_y = random.randint(0, img_h-1)
		random_x = random.randint(0, img_w-1)
		inputmat[random_y, random_x,:] = [0,0,0]

	return inputmat

def apply_gaussian_blur(inputmat, blur_val=1):
	print('image_augmentation_fns.apply_gaussian_blur.blur_val=',blur_val)
	val = blur_val * 2 + 1
	blur = cv2.GaussianBlur(inputmat, (val, val), 0)
	return blur

def apply_motion_blur(inputmat, size):
	print('image_augmentation_fns.apply_motion_blur')
	# generating the kernel
	kernel_motion_blur = np.zeros((size, size))
	kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
	kernel_motion_blur = kernel_motion_blur / size

	# applying the kernel to the input image
	blur = cv2.filter2D(inputmat, -1, kernel_motion_blur)
	return blur

def apply_low_res_by_resizing(inputmat, min_smaller_resize_ratio=0.5, max_smaller_resize_ratio=0.7,interpolation = cv2.INTER_CUBIC):
	print('image_augmentation_fns.apply_low_res_by_resizing')
	img_h, img_w, _ = inputmat.shape
	smaller_resize_ratio = random.uniform(min_smaller_resize_ratio, max_smaller_resize_ratio)
	small_imgmat = cv2.resize(inputmat, None, fx = smaller_resize_ratio, fy = smaller_resize_ratio, interpolation = interpolation)
	restored_imgmat = cv2.resize(small_imgmat, (img_w, img_h), interpolation = interpolation)
	return restored_imgmat

def apply_invert(inputmat):
	print('image_augmentation_fns.apply_invert')
	inverted_imgmat = 255 - inputmat
	return inverted_imgmat

def apply_jpeg_compression(img, min_intensity=60, max_intensity=100):
	print('image_augmentation_fns.apply_jpeg_compression')
	intensity = random.randint(min_intensity, max_intensity)
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), intensity]
	result, encimg = cv2.imencode('.jpg', img, encode_param)
	return cv2.imdecode(encimg, 1)

def apply_contrast_brightness_random_adjustment(img:np.ndarray, alpha:float=1.0, beta:int=0) -> np.ndarray:
	print('image_augmentation_fns.apply_contrast_brightness_random_adjustment')
	if not isinstance(beta, numbers.Real) or not isinstance(alpha, numbers.Real):
		raise TypeError

	beta = int(beta)
	assert alpha >= 1.0 and alpha <= 3.0, "the alpha value must be between 1.0 and 3.0"
	assert beta >= 0 and beta <= 100, "beta value must be between 0 and 100"
	return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

class AugmentationManager:
	def __init__(self):
		self.fn_bundle_list=[]
		self.fn_weight_list=[]

	def add_fn(self, fn, str_repr, probability_weight):
		assert probability_weight is not None
		assert probability_weight >0

		fn_bundle = (fn, str_repr)
		self.fn_bundle_list.append(fn_bundle)
		self.fn_weight_list.append(probability_weight)

	def random_pick_fn(self):

		selected_fn_bundle = random.choices(self.fn_bundle_list, weights=self.fn_weight_list, k=1)
		selected_fn_bundle = selected_fn_bundle[0]
		print("augmgr: {} selected".format(selected_fn_bundle[1]))
		return selected_fn_bundle[0]


class PipeLineAugmentationManager(AugmentationManager):

	def add_fn(self, fn, str_repr, probability):
		assert probability is not None
		assert probability >0
		assert probability <= 1.0
		assert fn is not None

		fn_bundle= (fn, str_repr)
		self.fn_bundle_list.append(fn_bundle)
		self.fn_weight_list.append(probability)

	def apply_augmentation(self, imgmat, font_size):
		for fn_bundle, probability in zip(self.fn_bundle_list, self.fn_weight_list):
			random_float = random.random()
			fn = fn_bundle[0]
			fn_str_repr = fn_bundle[1]
			if(font_size<22 and fn_str_repr=='ready_apply_low_res_by_resizing'):
				continue

			if random_float <= probability:
				imgmat = fn(imgmat)

		return imgmat

		