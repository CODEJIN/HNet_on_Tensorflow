# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_main_Dialog(object):
    def setupUi(self, main_Dialog):
        main_Dialog.setObjectName("main_Dialog")
        main_Dialog.resize(860, 334)
        self.structureSetup_Button = QtWidgets.QPushButton(main_Dialog)
        self.structureSetup_Button.setGeometry(QtCore.QRect(10, 10, 181, 35))
        self.structureSetup_Button.setObjectName("structureSetup_Button")
        self.exit_Button = QtWidgets.QPushButton(main_Dialog)
        self.exit_Button.setGeometry(QtCore.QRect(90, 290, 171, 35))
        self.exit_Button.setObjectName("exit_Button")
        self.patternSetup_Button = QtWidgets.QPushButton(main_Dialog)
        self.patternSetup_Button.setGeometry(QtCore.QRect(10, 50, 250, 35))
        self.patternSetup_Button.setObjectName("patternSetup_Button")
        self.processSetup_Button = QtWidgets.QPushButton(main_Dialog)
        self.processSetup_Button.setEnabled(False)
        self.processSetup_Button.setGeometry(QtCore.QRect(10, 90, 181, 35))
        self.processSetup_Button.setObjectName("processSetup_Button")
        self.learning_Button = QtWidgets.QPushButton(main_Dialog)
        self.learning_Button.setEnabled(False)
        self.learning_Button.setGeometry(QtCore.QRect(10, 210, 250, 35))
        self.learning_Button.setObjectName("learning_Button")
        self.macro_Button = QtWidgets.QPushButton(main_Dialog)
        self.macro_Button.setGeometry(QtCore.QRect(139, 250, 121, 35))
        self.macro_Button.setObjectName("macro_Button")
        self.about_Button = QtWidgets.QPushButton(main_Dialog)
        self.about_Button.setGeometry(QtCore.QRect(10, 290, 71, 35))
        self.about_Button.setObjectName("about_Button")
        self.state_GroupBox = QtWidgets.QGroupBox(main_Dialog)
        self.state_GroupBox.setGeometry(QtCore.QRect(270, 10, 581, 301))
        self.state_GroupBox.setObjectName("state_GroupBox")
        self.connection_Label = QtWidgets.QLabel(self.state_GroupBox)
        self.connection_Label.setGeometry(QtCore.QRect(390, 16, 100, 15))
        self.connection_Label.setObjectName("connection_Label")
        self.layer_Label = QtWidgets.QLabel(self.state_GroupBox)
        self.layer_Label.setGeometry(QtCore.QRect(200, 16, 100, 15))
        self.layer_Label.setObjectName("layer_Label")
        self.process_Label = QtWidgets.QLabel(self.state_GroupBox)
        self.process_Label.setGeometry(QtCore.QRect(200, 160, 100, 15))
        self.process_Label.setObjectName("process_Label")
        self.patternPack_Label = QtWidgets.QLabel(self.state_GroupBox)
        self.patternPack_Label.setGeometry(QtCore.QRect(10, 160, 100, 15))
        self.patternPack_Label.setObjectName("patternPack_Label")
        self.layer_ListWidget = QtWidgets.QListWidget(self.state_GroupBox)
        self.layer_ListWidget.setGeometry(QtCore.QRect(200, 30, 180, 120))
        self.layer_ListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.layer_ListWidget.setObjectName("layer_ListWidget")
        self.connection_ListWidget = QtWidgets.QListWidget(self.state_GroupBox)
        self.connection_ListWidget.setGeometry(QtCore.QRect(390, 30, 180, 120))
        self.connection_ListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.connection_ListWidget.setObjectName("connection_ListWidget")
        self.process_ListWidget = QtWidgets.QListWidget(self.state_GroupBox)
        self.process_ListWidget.setGeometry(QtCore.QRect(200, 174, 180, 120))
        self.process_ListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.process_ListWidget.setObjectName("process_ListWidget")
        self.patternPack_ListWidget = QtWidgets.QListWidget(self.state_GroupBox)
        self.patternPack_ListWidget.setGeometry(QtCore.QRect(10, 174, 180, 120))
        self.patternPack_ListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.patternPack_ListWidget.setObjectName("patternPack_ListWidget")
        self.learningSetup_Label = QtWidgets.QLabel(self.state_GroupBox)
        self.learningSetup_Label.setGeometry(QtCore.QRect(390, 160, 100, 15))
        self.learningSetup_Label.setObjectName("learningSetup_Label")
        self.learningSetup_ListWidget = QtWidgets.QListWidget(self.state_GroupBox)
        self.learningSetup_ListWidget.setGeometry(QtCore.QRect(390, 174, 181, 120))
        self.learningSetup_ListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.learningSetup_ListWidget.setObjectName("learningSetup_ListWidget")
        self.layer_Label_2 = QtWidgets.QLabel(self.state_GroupBox)
        self.layer_Label_2.setGeometry(QtCore.QRect(10, 16, 100, 15))
        self.layer_Label_2.setObjectName("layer_Label_2")
        self.configVariables_ListWidget = QtWidgets.QListWidget(self.state_GroupBox)
        self.configVariables_ListWidget.setGeometry(QtCore.QRect(10, 30, 181, 121))
        self.configVariables_ListWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.configVariables_ListWidget.setObjectName("configVariables_ListWidget")
        self.structureLock_Button = QtWidgets.QPushButton(main_Dialog)
        self.structureLock_Button.setEnabled(False)
        self.structureLock_Button.setGeometry(QtCore.QRect(200, 10, 61, 35))
        self.structureLock_Button.setObjectName("structureLock_Button")
        self.processLock_Button = QtWidgets.QPushButton(main_Dialog)
        self.processLock_Button.setEnabled(False)
        self.processLock_Button.setGeometry(QtCore.QRect(200, 90, 61, 35))
        self.processLock_Button.setObjectName("processLock_Button")
        self.weightAndBiasLoad_Button = QtWidgets.QPushButton(main_Dialog)
        self.weightAndBiasLoad_Button.setGeometry(QtCore.QRect(10, 170, 251, 35))
        self.weightAndBiasLoad_Button.setObjectName("weightAndBiasLoad_Button")
        self.copyright_Label = QtWidgets.QLabel(main_Dialog)
        self.copyright_Label.setGeometry(QtCore.QRect(270, 310, 581, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(200, 200, 200))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(200, 200, 200))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.copyright_Label.setPalette(palette)
        self.copyright_Label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.copyright_Label.setObjectName("copyright_Label")
        self.learningSetup_Button = QtWidgets.QPushButton(main_Dialog)
        self.learningSetup_Button.setEnabled(False)
        self.learningSetup_Button.setGeometry(QtCore.QRect(10, 130, 250, 35))
        self.learningSetup_Button.setObjectName("learningSetup_Button")
        self.modelSaveforMacro_Button = QtWidgets.QPushButton(main_Dialog)
        self.modelSaveforMacro_Button.setEnabled(False)
        self.modelSaveforMacro_Button.setGeometry(QtCore.QRect(10, 250, 121, 35))
        self.modelSaveforMacro_Button.setObjectName("modelSaveforMacro_Button")

        self.retranslateUi(main_Dialog)
        QtCore.QMetaObject.connectSlotsByName(main_Dialog)

    def retranslateUi(self, main_Dialog):
        _translate = QtCore.QCoreApplication.translate
        main_Dialog.setWindowTitle(_translate("main_Dialog", "HNet on Tensorflow 0.9.1.0"))
        self.structureSetup_Button.setText(_translate("main_Dialog", "Structure Setup"))
        self.exit_Button.setText(_translate("main_Dialog", "Exit"))
        self.patternSetup_Button.setText(_translate("main_Dialog", "Pattern Setup"))
        self.processSetup_Button.setText(_translate("main_Dialog", "Process Setup"))
        self.learning_Button.setText(_translate("main_Dialog", "Learning"))
        self.macro_Button.setText(_translate("main_Dialog", "Macro"))
        self.about_Button.setText(_translate("main_Dialog", "About"))
        self.state_GroupBox.setTitle(_translate("main_Dialog", "Status"))
        self.connection_Label.setText(_translate("main_Dialog", "Connection"))
        self.layer_Label.setText(_translate("main_Dialog", "Layer"))
        self.process_Label.setText(_translate("main_Dialog", "Process"))
        self.patternPack_Label.setText(_translate("main_Dialog", "Pattern Pack"))
        self.learningSetup_Label.setText(_translate("main_Dialog", "Learning Setup"))
        self.layer_Label_2.setText(_translate("main_Dialog", "Config Variables"))
        self.structureLock_Button.setText(_translate("main_Dialog", "Lock"))
        self.processLock_Button.setText(_translate("main_Dialog", "Lock"))
        self.weightAndBiasLoad_Button.setText(_translate("main_Dialog", "Weight && Bias Load"))
        self.copyright_Label.setText(_translate("main_Dialog", "Copyright(c) 2016-2017 Heejo You All rights reserved. "))
        self.learningSetup_Button.setText(_translate("main_Dialog", "Learning Setup"))
        self.modelSaveforMacro_Button.setText(_translate("main_Dialog", "Model Save\n"
"for Macro"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_Dialog = QtWidgets.QDialog()
    ui = Ui_main_Dialog()
    ui.setupUi(main_Dialog)
    main_Dialog.show()
    sys.exit(app.exec_())

