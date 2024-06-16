from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from image import resize_image, preprocess_image, show_cam_on_image, save_img
from cam import get_grad_cam, shapley_value
from PIL import Image
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Run GS-CAM.')
parser.add_argument('--samples', '-s', type=int, help='a number of samples', default=10000)
parser.add_argument('--batches', '-b', type=int, help='a number of batches', default=20)
parser.add_argument('--threshold', '-t', type=float, help='a threshold value', default=0.5)
parser.add_argument('--device', '-d', type=str, help='a type of device', default="cuda")
args = parser.parse_args()
NUM_SAMPLES = args.samples
NUM_BATCHES = args.batches
THRESHOLD = args.threshold
DEVICE = args.device


model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
model = model.to(DEVICE)
model.eval()

target_size = (224, 224)
filename = "bear"
path = f"imgs/{filename}.png"

image = np.array(Image.open(path).convert('RGB').resize(target_size))
image = np.float32(image) / 255
image = image[None, :]

input_tensor = preprocess_image(image)
input_tensor = input_tensor.to(DEVICE)

grad_values = get_grad_cam(model, input_tensor)
gs_values = shapley_value(model, input_tensor, device=DEVICE, grad_cam=grad_values, threshold=THRESHOLD, num_samples = NUM_SAMPLES, num_batches = NUM_BATCHES)

gs_cam = resize_image(gs_values, target_size)

gs_cam_img = show_cam_on_image(image[0], gs_cam[0], use_rgb=True)
save_img(gs_cam_img, "output.png")