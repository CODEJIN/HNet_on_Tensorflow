# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PatternSetup.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(661, 326)
        self.filePath_LineEdit = QtWidgets.QLineEdit(Dialog)
        self.filePath_LineEdit.setGeometry(QtCore.QRect(90, 220, 391, 20))
        self.filePath_LineEdit.setReadOnly(True)
        self.filePath_LineEdit.setObjectName("filePath_LineEdit")
        self.packName_LineEdit = QtWidgets.QLineEdit(Dialog)
        self.packName_LineEdit.setGeometry(QtCore.QRect(90, 250, 100, 20))
        self.packName_LineEdit.setObjectName("packName_LineEdit")
        self.delete_Button = QtWidgets.QPushButton(Dialog)
        self.delete_Button.setGeometry(QtCore.QRect(10, 170, 161, 31))
        self.delete_Button.setObjectName("delete_Button")
        self.insert_Button = QtWidgets.QPushButton(Dialog)
        self.insert_Button.setGeometry(QtCore.QRect(570, 220, 81, 51))
        self.insert_Button.setObjectName("insert_Button")
        self.filePath_Label = QtWidgets.QLabel(Dialog)
        self.filePath_Label.setGeometry(QtCore.QRect(10, 220, 70, 13))
        self.filePath_Label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.filePath_Label.setObjectName("filePath_Label")
        self.packName_Label = QtWidgets.QLabel(Dialog)
        self.packName_Label.setGeometry(QtCore.QRect(10, 250, 70, 13))
        self.packName_Label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.packName_Label.setObjectName("packName_Label")
        self.exit_Button = QtWidgets.QPushButton(Dialog)
        self.exit_Button.setGeometry(QtCore.QRect(240, 290, 200, 30))
        self.exit_Button.setObjectName("exit_Button")
        self.broswer_Button = QtWidgets.QPushButton(Dialog)
        self.broswer_Button.setGeometry(QtCore.QRect(490, 220, 75, 23))
        self.broswer_Button.setObjectName("broswer_Button")
        self.patternPack_ListWidget = QtWidgets.QListWidget(Dialog)
        self.patternPack_ListWidget.setGeometry(QtCore.QRect(10, 10, 161, 151))
        self.patternPack_ListWidget.setObjectName("patternPack_ListWidget")
        self.patternPackInformation_TextEdit = QtWidgets.QTextEdit(Dialog)
        self.patternPackInformation_TextEdit.setGeometry(QtCore.QRect(180, 10, 471, 192))
        self.patternPackInformation_TextEdit.setObjectName("patternPackInformation_TextEdit")
        self.frame_3 = QtWidgets.QFrame(Dialog)
        self.frame_3.setGeometry(QtCore.QRect(380, 250, 181, 21))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.fileTypeWide_RadioButton = QtWidgets.QRadioButton(self.frame_3)
        self.fileTypeWide_RadioButton.setGeometry(QtCore.QRect(10, 0, 51, 17))
        self.fileTypeWide_RadioButton.setChecked(True)
        self.fileTypeWide_RadioButton.setObjectName("fileTypeWide_RadioButton")
        self.fileTypeLong_RadioButton = QtWidgets.QRadioButton(self.frame_3)
        self.fileTypeLong_RadioButton.setGeometry(QtCore.QRect(70, 0, 51, 17))
        self.fileTypeLong_RadioButton.setObjectName("fileTypeLong_RadioButton")
        self.fileTypePickle_RadioButton = QtWidgets.QRadioButton(self.frame_3)
        self.fileTypePickle_RadioButton.setGeometry(QtCore.QRect(130, 0, 51, 17))
        self.fileTypePickle_RadioButton.setObjectName("fileTypePickle_RadioButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Pattern Setup"))
        self.delete_Button.setText(_translate("Dialog", "Delete"))
        self.insert_Button.setText(_translate("Dialog", "Insert"))
        self.filePath_Label.setText(_translate("Dialog", "File Path"))
        self.packName_Label.setText(_translate("Dialog", "Pack Name"))
        self.exit_Button.setText(_translate("Dialog", "Exit"))
        self.broswer_Button.setText(_translate("Dialog", "Broswer..."))
        self.patternPackInformation_TextEdit.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.fileTypeWide_RadioButton.setText(_translate("Dialog", "Wide"))
        self.fileTypeLong_RadioButton.setText(_translate("Dialog", "Long"))
        self.fileTypePickle_RadioButton.setText(_translate("Dialog", "Pickle"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

