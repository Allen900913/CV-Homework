from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from GUI import Ui_MainWindow
import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from vgg.vg import VGG19Model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MainWindow_controller(QtWidgets.QMainWindow):
    InferenceImg = None

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.LoadImg.clicked.connect(self.getImg)
        self.ui.ShowAugmentedImg.clicked.connect(self.showImg)
        self.ui.ShowModelStructure.clicked.connect(self.showModelStructure)
        self.ui.ShowAccuracyAndLoss.clicked.connect(self.showAccAndLoss)
        self.ui.Inference.clicked.connect(self.inference)

    def getImg(self):
        filename = QFileDialog.getOpenFileName()
        self.InferenceImg = filename[0]
        img = cv2.imread(self.InferenceImg)
        showimg = cv2.resize(img, (128, 128))
        cv2.imwrite("./out/cifar10show.png", showimg)
        pixmap = QPixmap("./out/cifar10show.png")
        self.ui.inferenceImg.setPixmap(pixmap)

    def showImg(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ])

        paths = os.listdir("./Q1_image/Q1_1")
        imgpaths = [f"./Q1_image/Q1_1/{i}" for i in paths]
        names = [i[:-4] for i in paths]

        augmented_images = [transform(Image.open(image_path)) for image_path in imgpaths]

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle("Augmented Image")

        for i, img in enumerate(augmented_images):
            row, col = divmod(i, 3)
            axs[row, col].imshow(img.permute(1, 2, 0))
            axs[row, col].title.set_text(names[i])

        plt.show()

    def showModelStructure(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG19Model().to(device)
        model.load_state_dict(torch.load('./vgg/vgg19_cifar10.pth', map_location=device))
        model.eval()
        print(model)

    def showAccAndLoss(self):
        imgAcc = cv2.imread("./vgg/accuracy.png")
        imgloss = cv2.imread("./vgg/loss.png")
        
        cv2.imshow('Acc Curve', imgAcc)
        cv2.imshow('Loss Curve', imgloss)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def inference(self):
        cifar10_class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        if not self.InferenceImg:
            print("Image not loaded. Please check the resource!")
            return

        img = cv2.imread(self.InferenceImg)
        self.ui.Predict.setText("Predicting...")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(img).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG19Model().to(device)
        model.load_state_dict(torch.load('./vgg/vgg19_cifar10.pth', map_location=device))
        model.eval()

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]

        max_index = probabilities.argmax()
        predLabel = cifar10_class_names[max_index]

        self.ui.Predict.setText(f"Predicted = {predLabel}")

        plt.bar(cifar10_class_names, probabilities)
        plt.show()
