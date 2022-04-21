from configparser import Interpolation
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
from architecture import CNN_LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = cv2.imread("data/image.jpeg")
image_2 = cv2.imread("data/image1.jpeg")

image = cv2.resize(image, (1280, 348), interpolation = cv2.INTER_LINEAR)
image_2 = cv2.resize(image_2, (1280,348), interpolation = cv2.INTER_LINEAR)

transform = transforms.ToTensor()

image = transform(image)
# image = image.unsqueeze(0)

image_2 = transform(image_2)
# image_2 = image_2.unsqueeze(0)

stacked_image = torch.stack(
    (image, image_2),
    dim = 0
)

stacked_image = stacked_image.to(device)

cnn_lstm = CNN_LSTM()

cnn_lstm.eval()
cnn_lstm = cnn_lstm.to(device)



x = cnn_lstm(stacked_image)

print("Final Shape - \n",x.shape)

print(x)
