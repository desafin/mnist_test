import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import cv2  # OpenCV 라이브러리 추가




# 모델 정의 (이미 정의한 모델을 사용합니다)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 모델 클래스 정의 후에 모델 인스턴스 생성
model = CNN()

# 모델 불러오기 (경로에 있는 모델 파일의 경로를 지정하세요)
model_path = 'your_model.pth'  # 모델 파일의 경로를 지정하세요
model.load_state_dict(torch.load(model_path))

# 모델을 GPU로 이동 (필요한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델을 평가 모드로 설정
model.eval()

# 이미지 파일이 있는 디렉토리 경로를 지정합니다.
image_dir = 'testing_img'  # 이미지 파일이 있는 디렉토리 경로를 지정하세요

# 디렉토리 내의 모든 이미지 파일을 가져옵니다.
image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

# 이미지를 불러와서 PyTorch 텐서로 변환합니다.
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # 이미지를 흑백으로 변환
                                transforms.ToTensor(),  # PyTorch 텐서로 변환
                                transforms.Normalize((0.5,), (0.5,))])  # 정규화

# 각 이미지 파일에 대해 추론을 수행하면서 이미지를 크게 표시합니다.
for image_file in image_files:
    image = Image.open(image_file).convert('L')  # 흑백으로 변환
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 및 GPU로 이동

    # 이미지를 모델에 전달하여 추론합니다.
    output = model(image)
    probabilities = torch.softmax(output, dim=1)  # 확률 분포 계산
    predicted_label = torch.argmax(output, 1).item()  # 가장 높은 확률을 가진 클래스

    # 이미지를 OpenCV를 사용하여 크게 표시하고 클래스별 확률을 표시합니다.
    img = cv2.imread(image_file)
    img = cv2.resize(img, (400, 400))  # 이미지 크기 조정
    cv2.putText(img, f"Predicted Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # 결과 이미지에 확률 정보 표시
    for i in range(probabilities.shape[1]):
        class_prob = probabilities[0, i].item()
        cv2.putText(img, f"Class {i}: {class_prob:.4f}", (10, 70 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    2)

    # 결과 이미지를 저장할 경로 및 파일 이름 생성
    output_path = "output_images/"
    output_filename = f"{output_path}{len(os.listdir(output_path))}.png"

    # 이미지 저장
    cv2.imwrite(output_filename, img)

    # 결과 이미지를 화면에 표시
    cv2.imshow(f"Image: {image_file}", img)

    # 창을 열어놓고 키 입력을 대기
    cv2.waitKey(0)

# OpenCV 창을 닫습니다.
cv2.destroyAllWindows()