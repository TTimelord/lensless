from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, pyqtSignal
import cv2
import sys
import threading


class myImageDisplayApp(QObject):
    # Define the custom signal
    # https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#the-pyqtslot-decorator
    signal_update_image = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Setup the seperate thread
        # https://stackoverflow.com/a/37694109/4988010
        self.thread = threading.Thread(target=self.run_app_widget_in_background)
        self.thread.daemon = True
        self.thread.start()

    def run_app_widget_in_background(self):
        self.app = QApplication(sys.argv)
        self.my_bg_qt_app = qtAppWidget(main_thread_object=self)
        self.app.exec_()

    def emit_image_update(self, pattern_file=None):
        # print('next image')
        self.signal_update_image.emit(pattern_file)


class qtAppWidget(QLabel):

    def __init__(self, main_thread_object):
        super().__init__()

        # Connect the singal to slot
        main_thread_object.signal_update_image.connect(self.update_image_from_file)

        self.setupGUI()

    def setupGUI(self):
        self.app = QApplication.instance()

        # Get avaliable screens/monitors
        # https://doc.qt.io/qt-5/qscreen.html
        # Get info on selected screen
        self.selected_screen = 1  # Select the desired monitor/screen

        self.screens_available = self.app.screens()
        print(self.screens_available)
        self.screen = self.screens_available[self.selected_screen]
        print
        self.screen_width = self.screen.size().width()
        self.screen_height = self.screen.size().height()

        # Create a black image for init
        self.pixmap = QPixmap(self.screen_width, self.screen_height)
        self.pixmap.fill(QColor('black'))

        # Create QLabel object
        self.app_widget = QLabel()

        # Varioius flags that can be applied to make displayed window frameless, fullscreen, etc...
        # https://doc.qt.io/qt-5/qt.html#WindowType-enum
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum

        self.app_widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.WindowStaysOnTopHint)
        # self.app_widget.setWindowFlags(Qt.FramelessWindowHint)

        # Hide mouse cursor
        self.app_widget.setCursor(Qt.BlankCursor)

        # self.app_widget.setGeometry(0, 0, self.screen_width, self.screen_height)            # Set the size of Qlabel to size of the screen
        self.app_widget.setGeometry(3840, 0, self.screen_width,
                                    self.screen_height)  # mlq: I have to set x to 3840 for displaying properly. The number cannot be used to move the picture because the window is set to be full screen below.

        self.app_widget.setWindowTitle('myImageDisplayApp')
        self.app_widget.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # https://doc.qt.io/qt-5/qt.html#AlignmentFlag-enum
        self.app_widget.setPixmap(self.pixmap)
        self.app_widget.show()

        # Set the screen on which widget is on
        self.app_widget.windowHandle().setScreen(self.screen)
        # Make full screen
        self.app_widget.showFullScreen()

    def update_image_from_file(self, pattern_file: str = None):
        print('Pattern file given: ', pattern_file)
        self.app_widget.clear()  # Clear all existing content of the QLabel
        pixmap = QPixmap(pattern_file)  # Update pixmap with desired image
        self.app_widget.setPixmap(pixmap)  # Show desired image on Qlabel

    def update_image_from_opencv(self, cvimg):
        height, width, depth = cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
        qpixmap = QPixmap(qimg)
        self.app_widget.setPixmap(qpixmap)
