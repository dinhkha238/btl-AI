from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QLabel, QPushButton,QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from tensorflow.keras.models import load_model
import numpy as np
import keras.utils as image

model = load_model('my_model_5.h5')
def predict_image(image_path):
    #Input image
    test_image = image.load_img(image_path,target_size=(200,200))
    #For show image
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    # Result array
    result = model.predict(test_image)
    if(result>=0.5):
        QMessageBox.information(None, 'Thông báo', 'Đây là con chó')
    else:
        QMessageBox.information(None, 'Thông báo', 'Đây là con mèo')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Phân loại chó mèo')
        self.setGeometry(100, 100, 700, 600)
        
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 400, 400)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setText('Chọn ảnh để phân loại')
        
        self.button = QPushButton('Chọn ảnh', self)
        self.button.setGeometry(200, 450, 100, 30)
        self.button.clicked.connect(self.select_image)
        
        self.result_label = QLabel(self)
        self.result_label.setGeometry(200, 10, 100, 30)
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        
    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.xpm *.jpg)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.label.setPixmap(pixmap)
            #set resized image
            self.label.setScaledContents(True)

            result = predict_image(file_name)
            self.result_label.setText(result)
            
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
