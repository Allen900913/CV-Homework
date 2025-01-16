from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)  # Resize to fit content

        # Main font setup
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        MainWindow.setFont(font)

        # Central widget setup
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        # Main layout - horizontal layout
        self.mainLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)

        # Left-side buttons layout
        self.buttonLayout = QtWidgets.QVBoxLayout()
        self.buttonLayout.setSpacing(10)

        self.LoadImg = QtWidgets.QPushButton("Load Image", self.centralwidget)
        self.buttonLayout.addWidget(self.LoadImg)

        self.ShowAugmentedImg = QtWidgets.QPushButton("1. Show Augmented Images", self.centralwidget)
        self.buttonLayout.addWidget(self.ShowAugmentedImg)

        self.ShowModelStructure = QtWidgets.QPushButton("2. Show Model Structure", self.centralwidget)
        self.buttonLayout.addWidget(self.ShowModelStructure)

        self.ShowAccuracyAndLoss = QtWidgets.QPushButton("3. Show Accuracy and Loss", self.centralwidget)
        self.buttonLayout.addWidget(self.ShowAccuracyAndLoss)

        self.Inference = QtWidgets.QPushButton("4. Inference", self.centralwidget)
        self.buttonLayout.addWidget(self.Inference)

        self.mainLayout.addLayout(self.buttonLayout)

        # Right-side image and label layout
        self.rightLayout = QtWidgets.QVBoxLayout()
        self.rightLayout.setSpacing(10)

        # Image display
        self.inferenceImg = QtWidgets.QLabel(self.centralwidget)
        self.inferenceImg.setFixedSize(300, 300)  # Set a fixed size for the image
        self.inferenceImg.setStyleSheet("background-color: white; border: 1px solid black;")
        self.inferenceImg.setAlignment(QtCore.Qt.AlignCenter)
        self.rightLayout.addWidget(self.inferenceImg)

        # Prediction text
        self.Predict = QtWidgets.QLabel("Predicted = ", self.centralwidget)
        font.setPointSize(10)
        font.setBold(True)
        self.Predict.setFont(font)
        self.Predict.setAlignment(QtCore.Qt.AlignCenter)
        self.rightLayout.addWidget(self.Predict)

        self.mainLayout.addLayout(self.rightLayout)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
