###############################################################################
# HNet GUI
# Copyright (C) 2016-2017  Heejo You
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
##############################################################################

from HNet_Enum import *;
from HNet_Core import *;
from HNet_UI import *;
import numpy as np;
import sys;
from enum import Enum;
from PyQt5 import QtCore, QtGui, QtWidgets;
import matplotlib;
matplotlib.use("Qt5Agg");
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas;
from matplotlib.figure import Figure;
import _thread as thread;
import tensorflow as tf;
from tensorflow.python.client import device_lib;
import _pickle as pickle;
from copy import deepcopy


class Progress_Display_Canvas(FigureCanvas):    #Refer: https://www.boxcontrol.net/embedding-matplotlib-plot-on-pyqt5-gui.html    
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)
        #self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    """ Data's x-Axis: Epoch
        Data's y-Axis: MSE, CE, SS, Activation, etc."""
    def Update_Figure(self, xTicks, averaged_Result_Array, yAxis):
        self.fig.clear();
        self.axes = self.fig.add_subplot(111);
        self.axes.hold(False);

        self.axes.plot(xTicks, averaged_Result_Array, 'black');
        self.axes.set_xlim([0, np.max(xTicks)]);
        self.axes.set_ylim(yAxis);
        self.draw();

    """ Data's x-Axis: Cycle
        Data's y-Axis: Epoch
        Data Visualization: MSE, CE, SS, Activation, etc."""
    def Update_Figure_using_Cycle(self, xTicks, averaged_Result_List, yAxis, max_Cycle):        
        self.fig.clear();
        self.axes = self.fig.add_subplot(111);
        self.axes.hold(False);

        imShow = self.axes.imshow(averaged_Result_List, interpolation='nearest', vmin=yAxis[0], vmax=yAxis[1], aspect='auto', cmap="coolwarm");
        self.fig.colorbar(imShow);        
        self.fig.gca().set_xticks(np.arange(max_Cycle + 1));
        self.fig.gca().set_yticks(np.arange(len(xTicks)));
        self.fig.gca().set_yticklabels(xTicks);

        self.draw();

class Weight_Display_Canvas(FigureCanvas):    #Refer: https://www.boxcontrol.net/embedding-matplotlib-plot-on-pyqt5-gui.html    
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)
        #self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    """ Data's x-Axis: Cycle
        Data's y-Axis: Epoch
        Data Visualization: MSE, CE, SS, Activation, etc."""
    def Update_Figure(self, fromLayerName, toLayerName, weightMatrix):        
        self.fig.clear();
        self.axes = self.fig.add_subplot(111);
        self.axes.hold(False);

        #imShow = self.axes.imshow(averaged_Result_List, interpolation='nearest', vmin=yAxis[0], vmax=yAxis[1], cmap="coolwarm");
        imShow = self.axes.imshow(weightMatrix, interpolation='nearest', aspect='auto', cmap="coolwarm");
        self.fig.colorbar(imShow);
        self.fig.gca().set_ylabel(fromLayerName);
        self.fig.gca().set_xlabel(toLayerName);

        self.draw();

class HNet_GUI: 
    def __init__(self):
        guiFont = QtGui.QFont();
        guiFont.setFamily('Arial')
        guiFont.setFixedPitch(True)
        if sys.platform == "win32":
            guiFont.setPointSize(9);
            #guiFont.setPointSize(4);
        elif sys.platform in ["linux", "linux2"]:
            guiFont.setPointSize(8);
        elif sys.platform == "darwin":
            guiFont.setPointSize(7);
        QtWidgets.QApplication.setFont(guiFont);

        self.simulator = HNet();
        app = QtWidgets.QApplication(sys.argv)        
        self.QT_Windows_Initialize();
        self.QT_Function_Initialize();
        self.QT_RegExp_Validator_Initialize();
        self.main_Dialog.show();
        
        sys.exit(app.exec_());

    def QT_Windows_Initialize(self):
        self.main_Dialog = QtWidgets.QDialog();
        self.structure_Setup_Dialog = QtWidgets.QDialog();
        self.pattern_Setup_Dialog = QtWidgets.QDialog();
        self.process_Setup_Dialog = QtWidgets.QDialog();
        self.learning_Setup_Dialog = QtWidgets.QDialog();
        self.learning_Dialog = QtWidgets.QDialog();
        self.macro_Dialog = QtWidgets.QDialog();
        self.about_Dialog = QtWidgets.QDialog();
        
        self.main_UI = Main.Ui_main_Dialog();        
        self.structure_Setup_UI = StructureSetup.Ui_Dialog();
        self.pattern_Setup_UI = PatternSetup.Ui_Dialog();
        self.process_Setup_UI = ProcessSetup.Ui_Dialog();
        self.learning_Setup_UI = LearningSetup.Ui_Dialog();
        self.learning_UI = Learning.Ui_Dialog();
        self.macro_UI = Macro.Ui_Dialog();
        self.about_UI = About.Ui_Dialog();
        
        self.main_UI.setupUi(self.main_Dialog);
        self.structure_Setup_UI.setupUi(self.structure_Setup_Dialog);
        self.pattern_Setup_UI.setupUi(self.pattern_Setup_Dialog);
        self.process_Setup_UI.setupUi(self.process_Setup_Dialog);
        self.learning_Setup_UI.setupUi(self.learning_Setup_Dialog);
        self.learning_UI.setupUi(self.learning_Dialog);
        self.macro_UI.setupUi(self.macro_Dialog);
        self.about_UI.setupUi(self.about_Dialog);        

    def QT_Function_Initialize(self):
        self.Main_UI_Function_Setup();
        self.Structure_Setup_UI_Function_Setup();
        self.Pattern_Setup_UI_Function_Setup();
        self.Process_Setup_UI_Function_Setup();
        self.Learning_Setup_UI_Function_Setup();
        self.Learning_UI_Function_Setup();
        self.Macro_Function_Setup();
        self.About_UI_Function_Setup();
            
    def QT_RegExp_Validator_Initialize(self):
        self.float_Validator = QtGui.QRegExpValidator(QtCore.QRegExp("[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?"));
        self.positive_Float_Validator = QtGui.QRegExpValidator(QtCore.QRegExp("([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?"));
        self.int_Validator = QtGui.QRegExpValidator(QtCore.QRegExp("[-+]?([0-9]+)?"));
        self.positive_Int_Validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[1-9][0-9]*$"));
        self.letter_Validator = QtGui.QRegExpValidator(QtCore.QRegExp("^[1-9A-Za-z_]+$"));

        self.Structure_Setup_UI_RegExp_Setup();
        self.Pattern_Setup_UI_RegExp_Setup();
        self.Process_Setup_UI_RegExp_Setup();
        self.Learning_Setup_UI_RegExp_Setup();
        self.Learning_UI_RegExp_Setup();
        self.Macro_RegExp_Setup();

    # Start Function Setup
    def Main_UI_Function_Setup(self):
        self.main_UI.structureSetup_Button.clicked.connect(self.Main_UI_structureSetup_Button_Clicked);
        self.main_UI.structureLock_Button.clicked.connect(self.Main_UI_structureLock_Button_Clicked);
        self.main_UI.weightAndBiasLoad_Button.clicked.connect(self.Main_UI_weightAndBiasLoad_Button_Clicked);
        self.main_UI.patternSetup_Button.clicked.connect(self.Main_UI_patternSetup_Button_Clicked);
        self.main_UI.processSetup_Button.clicked.connect(self.Main_UI_processSetup_Button_Clicked);
        self.main_UI.processLock_Button.clicked.connect(self.Main_UI_processLock_Button_Clicked);
        self.main_UI.learningSetup_Button.clicked.connect(self.Main_UI_learningSetup_Button_Clicked);
        self.main_UI.learning_Button.clicked.connect(self.Main_UI_learning_Button_Clicked);
        self.main_UI.modelSaveforMacro_Button.clicked.connect(self.Main_UI_modelSaveforMacro_Button_Clicked);
        self.main_UI.macro_Button.clicked.connect(self.Main_UI_macro_Button_Clicked);        
        self.main_UI.about_Button.clicked.connect(self.Main_UI_about_Button_Clicked);        
        self.main_UI.exit_Button.clicked.connect(self.Main_UI_exit_Button_Clicked);

        self.Main_UI_Simulator_Changed();

    def Structure_Setup_UI_Function_Setup(self):
        self.structure_Setup_UI.summit_Button.clicked.connect(self.Structure_Setup_UI_summit_Button_Clicked);
        self.structure_Setup_UI.structureBPMake_Button.clicked.connect(self.Structure_Setup_UI_structureBPMake_Button_Clicked);
        self.structure_Setup_UI.structureBPTTMake_Button.clicked.connect(self.Structure_Setup_UI_structureBPTTMake_Button_Clicked);
        self.structure_Setup_UI.structureSRNMake_Button.clicked.connect(self.Structure_Setup_UI_structureSRNMake_Button_Clicked);
        self.structure_Setup_UI.layerAdd_Button.clicked.connect(self.Structure_Setup_UI_layerAdd_Button_Clicked);
        self.structure_Setup_UI.layerDelete_Button.clicked.connect(self.Structure_Setup_UI_layerDelete_Button_Clicked);
        self.structure_Setup_UI.connectionAdd_Button.clicked.connect(self.Structure_Setup_UI_connectionAdd_Button_Clicked);
        self.structure_Setup_UI.connectionDelete_Button.clicked.connect(self.Structure_Setup_UI_connectionDelete_Button_Clicked);
        self.structure_Setup_UI.structureSave_Button.clicked.connect(self.Structure_Setup_UI_structureSave_Button_Clicked);
        self.structure_Setup_UI.structureLoad_Button.clicked.connect(self.Structure_Setup_UI_structureLoad_Button_Clicked);        
        self.structure_Setup_UI.exit_Button.clicked.connect(self.Structure_Setup_UI_exit_Button_Clicked);

    def Pattern_Setup_UI_Function_Setup(self):
        self.pattern_Setup_UI.broswer_Button.clicked.connect(self.Pattern_Setup_UI_broswer_Button_Clicked);
        self.pattern_Setup_UI.insert_Button.clicked.connect(self.Pattern_Setup_UI_insert_Button_Clicked);
        self.pattern_Setup_UI.delete_Button.clicked.connect(self.Pattern_Setup_UI_delete_Button_Clicked);
        self.pattern_Setup_UI.patternPack_ListWidget.currentItemChanged.connect(self.Pattern_Setup_UI_patternPack_ListWidget_Current_Item_Changed); 
        self.pattern_Setup_UI.exit_Button.clicked.connect(self.Pattern_Setup_UI_exit_Button_Clicked);
    
    def Process_Setup_UI_Function_Setup(self):
        self.process_Setup_UI.processMaking_Button.clicked.connect(self.Process_Setup_UI_processMaking_Button_Clicked);
        self.process_Setup_UI.processModify_Button.clicked.connect(self.Process_Setup_UI_processModify_Button_Clicked);
        self.process_Setup_UI.processEnd_Button.clicked.connect(self.Process_Setup_UI_processEnd_Button_Clicked);
        self.process_Setup_UI.process_ListWidget.currentItemChanged.connect(self.Process_Setup_UI_process_ListWidget_Current_Item_Changed);
        self.process_Setup_UI.processDelete_Button.clicked.connect(self.Process_Setup_UI_processDelete_Button_Clicked);
        self.process_Setup_UI.layerOn_Button.clicked.connect(self.Process_Setup_UI_layerOn_Button_Clicked);
        self.process_Setup_UI.layerOff_Button.clicked.connect(self.Process_Setup_UI_layerOff_Button_Clicked);
        self.process_Setup_UI.layerDamage_Button.clicked.connect(self.Process_Setup_UI_layerDamage_Button_Clicked);
        self.process_Setup_UI.connectionOn_Button.clicked.connect(self.Process_Setup_UI_connectionOn_Button_Clicked);
        self.process_Setup_UI.connectionOff_Button.clicked.connect(self.Process_Setup_UI_connectionOff_Button_Clicked);
        self.process_Setup_UI.connectionDamage_Button.clicked.connect(self.Process_Setup_UI_connectionDamage_Button_Clicked);
        self.process_Setup_UI.orderDelete_Button.clicked.connect(self.Process_Setup_UI_orderDelete_Button_Clicked);
        self.process_Setup_UI.orderUp_Button.clicked.connect(self.Process_Setup_UI_orderUp_Button_Clicked);
        self.process_Setup_UI.orderDown_Button.clicked.connect(self.Process_Setup_UI_orderDown_Button_Clicked);
        self.process_Setup_UI.bpInputLayer_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_bpInputLayer_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.bpHiddenLayer_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_bpHiddenLayer_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.bpTrainingApply_Button.clicked.connect(self.Process_Setup_UI_bpTrainingApply_Button_Clicked);
        self.process_Setup_UI.bpTestApply_Button.clicked.connect(self.Process_Setup_UI_bpTestApply_Button_Clicked);
        self.process_Setup_UI.bpttTrainingApply_Button.clicked.connect(self.Process_Setup_UI_bpttTrainingApply_Button_Clicked);
        self.process_Setup_UI.bpttTestApply_Button.clicked.connect(self.Process_Setup_UI_bpttTestApply_Button_Clicked);
        self.process_Setup_UI.srnInputLayer_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_srnInputLayer_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.srnContextLayer_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_srnContextLayer_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.srnHiddenLayer_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_srnHiddenLayer_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.srnTrainingApply_Button.clicked.connect(self.Process_Setup_UI_srnTrainingApply_Button_Clicked);
        self.process_Setup_UI.srnTestApply_Button.clicked.connect(self.Process_Setup_UI_srnTestApply_Button_Clicked);
        self.process_Setup_UI.linearForwardAdd_Button.clicked.connect(self.Process_Setup_UI_linearForwardAdd_Button_Clicked);
        self.process_Setup_UI.linearForwardDelete_Button.clicked.connect(self.Process_Setup_UI_linearForwardDelete_Button_Clicked);
        self.process_Setup_UI.linearForwardApply_Button.clicked.connect(self.Process_Setup_UI_linearForwardApply_Button_Clicked);
        self.process_Setup_UI.linearBackwardAdd_Button.clicked.connect(self.Process_Setup_UI_linearBackwardAdd_Button_Clicked);
        self.process_Setup_UI.linearBackwardDelete_Button.clicked.connect(self.Process_Setup_UI_linearBackwardDelete_Button_Clicked);
        self.process_Setup_UI.linearBackwardApply_Button.clicked.connect(self.Process_Setup_UI_linearBackwardApply_Button_Clicked);
        self.process_Setup_UI.linearEndInitialize_Button.clicked.connect(self.Process_Setup_UI_linearEndInitialize_Button_Clicked);
        self.process_Setup_UI.customEndInitialize_Button.clicked.connect(self.Process_Setup_UI_customEndInitialize_Button_Clicked);
        self.process_Setup_UI.customInputLayerActivationInsert_Button.clicked.connect(self.Process_Setup_UI_customInputLayerActivationInsert_Button_Clicked);
        self.process_Setup_UI.customLayerActivationSend_Button.clicked.connect(self.Process_Setup_UI_customLayerActivationSend_Button_Clicked);
        self.process_Setup_UI.customLayerActivationCalculationSigmoid_Button.clicked.connect(self.Process_Setup_UI_customLayerActivationCalculationSigmoid_Button_Clicked);
        self.process_Setup_UI.customLayerActivationCalculationSoftmax_Button.clicked.connect(self.Process_Setup_UI_customLayerActivationCalculationSoftmax_Button_Clicked);
        self.process_Setup_UI.customLayerActivationCalculationReLU_Button.clicked.connect(self.Process_Setup_UI_customLayerActivationCalculationReLU_Button_Clicked);
        self.process_Setup_UI.customLayerInitialize_Button.clicked.connect(self.Process_Setup_UI_customLayerInitialize_Button_Clicked);
        self.process_Setup_UI.customOutputLayerErrorCalculationSigmoid_Button.clicked.connect(self.Process_Setup_UI_customOutputLayerErrorCalculationSigmoid_Button_Clicked);
        self.process_Setup_UI.customOutputLayerErrorCalculationSoftmax_Button.clicked.connect(self.Process_Setup_UI_customOutputLayerErrorCalculationSoftmax_Button_Clicked);
        self.process_Setup_UI.customHiddenLayerErrorCalculationSigmoid_Button.clicked.connect(self.Process_Setup_UI_customHiddenLayerErrorCalculationSigmoid_Button_Clicked);
        self.process_Setup_UI.customHiddenLayerErrorCalculationReLU_Button.clicked.connect(self.Process_Setup_UI_customHiddenLayerErrorCalculationReLU_Button_Clicked);
        self.process_Setup_UI.customErrorSend_Button.clicked.connect(self.Process_Setup_UI_customErrorSend_Button_Clicked);
        self.process_Setup_UI.customActivationExtract_Button.clicked.connect(self.Process_Setup_UI_customActivationExtract_Button_Clicked); 
        self.process_Setup_UI.customBiasRenewal_Button.clicked.connect(self.Process_Setup_UI_customBiasRenewal_Button_Clicked);
        self.process_Setup_UI.customWeightRenewal_Button.clicked.connect(self.Process_Setup_UI_customWeightRenewal_Button_Clicked);
        self.process_Setup_UI.customLayerDuplicate_Button.clicked.connect(self.Process_Setup_UI_customLayerDuplicate_Button_Clicked);
        self.process_Setup_UI.customConnectionDuplicate_Button.clicked.connect(self.Process_Setup_UI_customConnectionDuplicate_Button_Clicked);
        self.process_Setup_UI.customTransposedConnectionDuplicate_Button.clicked.connect(self.Process_Setup_UI_customTransposedConnectionDuplicate_Button_Clicked);
        self.process_Setup_UI.customBiasEqualizationAdd_Button.clicked.connect(self.Process_Setup_UI_customBiasEqualizationAdd_Button_Clicked);
        self.process_Setup_UI.customBiasEqualizationDelete_Button.clicked.connect(self.Process_Setup_UI_customBiasEqualizationDelete_Button_Clicked);
        self.process_Setup_UI.customBiasEqualization_Button.clicked.connect(self.Process_Setup_UI_customBiasEqualization_Button_Clicked);
        self.process_Setup_UI.customWeightEqualizationConnectionAdd_Button.clicked.connect(self.Process_Setup_UI_customWeightEqualizationConnectionAdd_Button_Clicked);
        self.process_Setup_UI.customWeightEqualizationConnectionDelete_Button.clicked.connect(self.Process_Setup_UI_customWeightEqualizationConnectionDelete_Button_Clicked);
        self.process_Setup_UI.customWeightEqualization_Button.clicked.connect(self.Process_Setup_UI_customWeightEqualization_Button_Clicked);
        self.process_Setup_UI.customCycleMaker_Button.clicked.connect(self.Process_Setup_UI_customCycleMaker_Button_Clicked);
        self.process_Setup_UI.customUniformRandomActivationInsert_Button.clicked.connect(self.Process_Setup_UI_customUniformRandomActivationInsert_Button_Clicked);
        self.process_Setup_UI.customNormalRandomActivationInsert_Button.clicked.connect(self.Process_Setup_UI_customNormalRandomActivationInsert_Button_Clicked);
        self.process_Setup_UI.customLayer1_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_customLayer1_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.customLayer2_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_customLayer2_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.customConnection1_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_customConnection1_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.customConnection2_ComboBox.currentIndexChanged.connect(self.Process_Setup_UI_customConnection2_ComboBox_Current_Index_Changed);
        self.process_Setup_UI.processSave_Button.clicked.connect(self.Process_Setup_UI_processSave_Button_Clicked);
        self.process_Setup_UI.processLoad_Button.clicked.connect(self.Process_Setup_UI_processLoad_Button_Clicked);
        self.process_Setup_UI.exit_Button.clicked.connect(self.Process_Setup_UI_exit_Button_Clicked);

    def Learning_Setup_UI_Function_Setup(self):
        self.learning_Setup_UI.learningSetupMaking_Button.clicked.connect(self.Learning_Setup_UI_learningSetupMaking_Button_Clicked);
        self.learning_Setup_UI.learningSetupModify_Button.clicked.connect(self.Learning_Setup_UI_learningSetupModify_Button_Clicked);
        self.learning_Setup_UI.learningSetupEnd_Button.clicked.connect(self.Learning_Setup_UI_learningSetupEnd_Button_Clicked);
        self.learning_Setup_UI.learningSetup_ListWidget.currentItemChanged.connect(self.Learning_Setup_UI_learningSetup_ListWidget_Current_Item_Changed);
        self.learning_Setup_UI.learningSetupDelete_Button.clicked.connect(self.Learning_Setup_UI_learningSetupDelete_Button_Clicked);
        self.learning_Setup_UI.learningSetupUp_Button.clicked.connect(self.Learning_Setup_UI_learningSetupUp_Button_Clicked);
        self.learning_Setup_UI.learningSetupDown_Button.clicked.connect(self.Learning_Setup_UI_learningSetupDown_Button_Clicked);
        self.learning_Setup_UI.trainingPatternMatchingMaking_Button.clicked.connect(self.Learning_Setup_UI_trainingPatternMatchingMaking_Button_Clicked);
        self.learning_Setup_UI.trainingPatternMatchingEnd_Button.clicked.connect(self.Learning_Setup_UI_trainingPatternMatchingEnd_Button_Clicked);
        self.learning_Setup_UI.trainingPatternMatching_ListWidget.currentItemChanged.connect(self.Learning_Setup_UI_trainingPatternMatching_ListWidget_Current_Item_Changed);        
        self.learning_Setup_UI.trainingPatternMatchingDelete_Button.clicked.connect(self.Learning_Setup_UI_trainingPatternMatchingDelete_Button_Clicked);
        self.learning_Setup_UI.trainingPatternMatchingUp_Button.clicked.connect(self.Learning_Setup_UI_trainingPatternMatchingUp_Button_Clicked);
        self.learning_Setup_UI.trainingPatternMatchingDown_Button.clicked.connect(self.Learning_Setup_UI_trainingPatternMatchingDown_Button_Clicked);
        self.learning_Setup_UI.trainingPatternPack_ComboBox.currentIndexChanged.connect(self.Learning_Setup_UI_trainingPatternPack_ComboBox_Current_Index_Changed);
        self.learning_Setup_UI.trainingProcess_ComboBox.currentIndexChanged.connect(self.Learning_Setup_UI_trainingProcess_ComboBox_Current_Index_Changed);
        self.learning_Setup_UI.trainingAutoAssign_Button.clicked.connect(self.Learning_Setup_UI_trainingAutoAssign_Button_Clicked);
        self.learning_Setup_UI.trainingPatternToOrderAssign_Button.clicked.connect(self.Learning_Setup_UI_trainingPatternToOrderAssign_Button_Clicked);
        self.learning_Setup_UI.trainingPatternToOrderDelete_Button.clicked.connect(self.Learning_Setup_UI_trainingPatternToOrderDelete_Button_Clicked);
        self.learning_Setup_UI.testPatternMatchingMaking_Button.clicked.connect(self.Learning_Setup_UI_testPatternMatchingMaking_Button_Clicked);
        self.learning_Setup_UI.testPatternMatchingEnd_Button.clicked.connect(self.Learning_Setup_UI_testPatternMatchingEnd_Button_Clicked);
        self.learning_Setup_UI.testPatternMatching_ListWidget.currentItemChanged.connect(self.Learning_Setup_UI_testPatternMatching_ListWidget_Current_Item_Changed);
        self.learning_Setup_UI.testPatternMatchingDelete_Button.clicked.connect(self.Learning_Setup_UI_testPatternMatchingDelete_Button_Clicked);
        self.learning_Setup_UI.testPatternPack_ComboBox.currentIndexChanged.connect(self.Learning_Setup_UI_testPatternPack_ComboBox_Current_Index_Changed);
        self.learning_Setup_UI.testProcess_ComboBox.currentIndexChanged.connect(self.Learning_Setup_UI_testProcess_ComboBox_Current_Index_Changed);
        self.learning_Setup_UI.testAutoAssign_Button.clicked.connect(self.Learning_Setup_UI_testAutoAssign_Button_Clicked);
        self.learning_Setup_UI.extractDataType_ComboBox.currentIndexChanged.connect(self.Learning_Setup_UI_extractDataType_ComboBox_Current_Index_Changed);
        self.learning_Setup_UI.testPatternToOrderAssign_Button.clicked.connect(self.Learning_Setup_UI_testPatternToOrderAssign_Button_Clicked);
        self.learning_Setup_UI.testPatternToOrderDelete_Button.clicked.connect(self.Learning_Setup_UI_testPatternToOrderDelete_Button_Clicked);
        self.learning_Setup_UI.extractDataAssign_Button.clicked.connect(self.Learning_Setup_UI_extractDataAssign_Button_Clicked);
        self.learning_Setup_UI.extractDataDelete_Button.clicked.connect(self.Learning_Setup_UI_extractDataDelete_Button_Clicked);
        self.learning_Setup_UI.save_Button.clicked.connect(self.Learning_Setup_UI_save_Button_Clicked);
        self.learning_Setup_UI.load_Button.clicked.connect(self.Learning_Setup_UI_load_Button_Clicked);
        self.learning_Setup_UI.exit_Button.clicked.connect(self.Learning_Setup_UI_exit_Button_Clicked);

    def Learning_UI_Function_Setup(self):
        self.learning_UI.start_Button.clicked.connect(self.Learning_UI_start_Button_Clicked);
        self.learning_UI.pause_Button.clicked.connect(self.Learning_UI_pause_Button_Clicked);
        self.learning_UI.testResultSave_Button.clicked.connect(self.Learning_UI_testResultSave_Button_Clicked);
        self.learning_UI.weightAndBiasSave_Button.clicked.connect(self.Learning_UI_weightAndBiasSave_Button_Clicked);
        self.learning_UI.weight_Display_Button.clicked.connect(self.Learning_UI_weight_Display_Button_Clicked);
        self.learning_UI.result_Display_Button.clicked.connect(self.Learning_UI_result_Display_Button_Clicked);
        self.learning_UI.exit_Button.clicked.connect(self.Learning_UI_exit_Button_Clicked);

        self.learning_UI.result_Graph = Progress_Display_Canvas(self.learning_UI.graph_Widget, width=5, height=3, dpi=50);
        self.learning_UI.progressGraphLayout.addWidget(self.learning_UI.result_Graph);

        self.learning_UI.weight_Graph = Weight_Display_Canvas(self.learning_UI.graph_Widget, width=5, height=3, dpi=50);
        self.learning_UI.weightGraphLayout.addWidget(self.learning_UI.weight_Graph);

    def Macro_Function_Setup(self):
        self.macro_UI.baseModelDataFileBroswer_Button.clicked.connect(self.Macro_UI_baseModelDataFileBroswer_Button_Clicked);
        self.macro_UI.baseModelDataFileLoad_Button.clicked.connect(self.Macro_UI_baseModelDataFileLoad_Button_Clicked);
        self.macro_UI.macroAdd_Button.clicked.connect(self.Macro_UI_macroAdd_Button_Clicked);
        self.macro_UI.macroDelete_Button.clicked.connect(self.Macro_UI_macroDelete_Button_Clicked);
        self.macro_UI.macro_ListWidget.currentItemChanged.connect(self.Macro_UI_macro_ListWidget_Current_Item_Changed);
        self.macro_UI.layerSizeAdd_Button.clicked.connect(self.Macro_UI_layerSizeAdd_Button_Clicked);
        self.macro_UI.regularMultiLayerSizeAdd_Button.clicked.connect(self.Macro_UI_regularMultiLayerSizeAdd_Button_Clicked);
        self.macro_UI.irregularMultiLayerSizeAdd_Button.clicked.connect(self.Macro_UI_irregularMultiLayerSizeAdd_Button_Clicked);
        self.macro_UI.learningRateAdd_Button.clicked.connect(self.Macro_UI_learningRateAdd_Button_Clicked);
        self.macro_UI.initialWeightSDAdd_Button.clicked.connect(self.Macro_UI_initialWeightSDAdd_Button_Clicked);
        self.macro_UI.layerDamageSDAdd_Button.clicked.connect(self.Macro_UI_layerDamageSDAdd_Button_Clicked);
        self.macro_UI.connectionDamageSDAdd_Button.clicked.connect(self.Macro_UI_connectionDamageSDAdd_Button_Clicked);
        self.macro_UI.modifyFactorDelete_Button.clicked.connect(self.Macro_UI_modifyFactorDelete_Button_Clicked);
        self.macro_UI.learning_Button.clicked.connect(self.Macro_UI_learning_Button_Clicked);
        self.macro_UI.exit_Button.clicked.connect(self.Macro_UI_exit_Button_Clicked);

    def About_UI_Function_Setup(self):
        self.about_UI.exit_Button.clicked.connect(self.About_Setup_UI_exit_Button_Clicked);
    
    # End Function Setup
    
    # Start RegExp Setup
    def Structure_Setup_UI_RegExp_Setup(self):
        self.structure_Setup_UI.momentum_LineEdit.setValidator(self.positive_Float_Validator);
        self.structure_Setup_UI.learningRate_LineEdit.setValidator(self.positive_Float_Validator);
        self.structure_Setup_UI.decayRate_LineEdit.setValidator(self.positive_Float_Validator);
        self.structure_Setup_UI.initalWeightSD_LineEdit.setValidator(self.positive_Float_Validator);
        self.structure_Setup_UI.structureBPInputLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureBPHiddenLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureBPOutputLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureBPTTInputLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureBPTTHiddenLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureBPTTOutputLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureBPTTTick_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureSRNInputLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureSRNHiddenLayerSize_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.structureSRNOutputLayerSize_LineEdit.setValidator(self.positive_Int_Validator);        
        self.structure_Setup_UI.layerName_LineEdit.setValidator(self.letter_Validator);
        self.structure_Setup_UI.layerUnit_LineEdit.setValidator(self.positive_Int_Validator);
        self.structure_Setup_UI.connectionName_LineEdit.setValidator(self.letter_Validator);

    def Pattern_Setup_UI_RegExp_Setup(self):
        self.pattern_Setup_UI.packName_LineEdit.setValidator(self.letter_Validator);

    def Process_Setup_UI_RegExp_Setup(self):
        self.process_Setup_UI.processName_LineEdit.setValidator(self.letter_Validator);
        self.process_Setup_UI.layerDamageSD_LineEdit.setValidator(self.positive_Float_Validator);
        self.process_Setup_UI.connectionDamageSD_LineEdit.setValidator(self.positive_Float_Validator);
        self.process_Setup_UI.bpttInputLayer_LineEdit.setValidator(self.letter_Validator);
        self.process_Setup_UI.bpttHiddenLayer_LineEdit.setValidator(self.letter_Validator);
        self.process_Setup_UI.bpttTick_LineEdit.setValidator(self.positive_Int_Validator);
        self.process_Setup_UI.srnMaxCycle_LineEdit.setValidator(self.positive_Int_Validator);
        self.process_Setup_UI.customRandomActivationCriteria_LineEdit.setValidator(self.positive_Float_Validator);

    def Learning_Setup_UI_RegExp_Setup(self):
        self.learning_Setup_UI.learningSetupName_LineEdit.setValidator(self.letter_Validator);
        self.learning_Setup_UI.trainingEpoch_LineEdit.setValidator(self.positive_Int_Validator);
        self.learning_Setup_UI.testTiming_LineEdit.setValidator(self.positive_Int_Validator);
        self.learning_Setup_UI.miniBatchSize_LineEdit.setValidator(self.positive_Int_Validator);

    def Learning_UI_RegExp_Setup(self):
        self.learning_UI.yAxisMin_LineEdit.setValidator(self.float_Validator);
        self.learning_UI.yAxisMax_LineEdit.setValidator(self.float_Validator);

    def Macro_RegExp_Setup(self):
        self.macro_UI.layerSizeFrom_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.layerSizeTo_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.layerSizeStep_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.regularMultiLayerSizeLayerPrefix_LineEdit.setValidator(self.letter_Validator);
        self.macro_UI.regularMultiLayerSizeMaxSuffix_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.regularMultiLayerSizeFrom_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.regularMultiLayerSizeTo_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.regularMultiLayerSizeStep_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.irregularMultiLayerSizeLayer_LineEdit.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("^([1-9A-Za-z_]+[ ])+$")));
        self.macro_UI.irregularMultiLayerSizeFrom_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.irregularMultiLayerSizeTo_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.irregularMultiLayerSizeStep_LineEdit.setValidator(self.positive_Int_Validator);
        self.macro_UI.learningRateFrom_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.learningRateTo_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.learningRateStep_LineEdit.setValidator(self.positive_Float_Validator);        
        self.macro_UI.initialWeightSDFrom_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.initialWeightSDTo_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.initialWeightSDStep_LineEdit.setValidator(self.positive_Float_Validator);        
        self.macro_UI.layerDamageSDFrom_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.layerDamageSDTo_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.layerDamageSDStep_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.connectionDamageSDFrom_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.connectionDamageSDTo_LineEdit.setValidator(self.positive_Float_Validator);
        self.macro_UI.connectionDamageSDStep_LineEdit.setValidator(self.positive_Float_Validator);
    # End RegExp Setup

    # Start Main UI Functions
    def Main_UI_structureSetup_Button_Clicked(self):
        self.structure_Setup_Dialog.show();
        self.Structure_Setup_UI_Structure_Changed();
        self.main_Dialog.hide();

    def Main_UI_structureLock_Button_Clicked(self):
        reply = QtWidgets.QMessageBox.question(None, 'Notice', "Model structure will lock. Do you want proceed?");
        if reply == QtWidgets.QMessageBox.Yes:
            self.main_UI.structureSetup_Button.setEnabled(False);
            self.main_UI.structureLock_Button.setEnabled(False);
            self.main_UI.weightAndBiasLoad_Button.setEnabled(True);
            self.main_UI.processSetup_Button.setEnabled(True);
            self.simulator.Weight_and_Bias_Setup();
    
    def Main_UI_weightAndBiasLoad_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        
        file_Path = new_FileDialog.getOpenFileName(filter="Model File for HNet (*.HNet_Model)")[0]
        if not file_Path == "":
            if not self.simulator.WeightAndBias_Load(file_Path):
                QtWidgets.QMessageBox.warning(None, 'Warning!', "This data file is not compatible with current structure.");

    def Main_UI_patternSetup_Button_Clicked(self):
        self.pattern_Setup_Dialog.show();
        self.Pattern_Setup_UI_Pattern_Pack_Changed();
        self.main_Dialog.hide();
    def Main_UI_processSetup_Button_Clicked(self):
        self.process_Setup_Dialog.show();
        self.current_Process_Order_List = [];
        self.current_Process_Layer_Control_Dict = {};
        self.current_Process_Connection_Control_Dict = {};
        self.current_Process_Linear_Forward_List = [];
        self.current_Process_Linear_Backward_List = [];
        self.current_Process_Custom_Bias_Equalization_Layer_List = [];
        self.current_Process_Custom_Weight_Equalization_Connection_List = [];
        self.Process_Setup_UI_ComboBox_Add_Item();
        self.Process_Setup_UI_Process_Changed();
        self.main_Dialog.hide();
    
    def Main_UI_processLock_Button_Clicked(self):
        reply = QtWidgets.QMessageBox.question(None, 'Notice', "Process will lock. Do you want proceed?");
        if reply == QtWidgets.QMessageBox.Yes:
            self.main_UI.processSetup_Button.setEnabled(False);
            self.main_UI.processLock_Button.setEnabled(False);
            self.main_UI.learningSetup_Button.setEnabled(True);                    
            self.simulator.Process_To_Tensor();

    def Main_UI_learningSetup_Button_Clicked(self):
        self.learning_Setup_Dialog.show();
        self.current_Training_Matching_List = []; #The List of Pattern_Matching
        self.current_Test_Matching_List = []; #The List of Pattern_Matching
        self.current_Training_Matching_Information = {};
        self.current_Training_Matching_Information["Assign"] = {};
        self.current_Test_Matching_Information = {};
        self.current_Test_Matching_Information["Assign"] = {};
        self.current_Test_Matching_Information["Extract_Data"] = [];
        self.Learning_Setup_UI_ComboBox_Add_Item();
        self.Learning_Setup_UI_Learning_Setup_Changed();
        self.main_Dialog.hide();
    def Main_UI_learning_Button_Clicked(self):
        self.learning_UI.macro_Label.hide();
        self.learning_UI.macro_LineEdit.hide();

        self.learning_Dialog.show();        
        self.Learning_UI_ComboBox_Add_Item();
        self.main_Dialog.hide();

    def Main_UI_macro_Button_Clicked(self):
        self.macro_Dialog.show();
        self.main_Dialog.hide();
    def Main_UI_modelSaveforMacro_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getSaveFileName(filter="Structure File for HNet (*.HNetGUI_Model_Data_for_Macro)")[0]
        if file_Path == "":
            return;

        save_Dict = {};
        save_Dict["Config_Dict"] = self.simulator.config_Variables_Dict;
        save_Dict["Layer_Dict"] = self.simulator.layer_Information_Dict;
        save_Dict["Connection_Dict"] = self.simulator.connection_Information_Dict;
        save_Dict["Pattern_Pack_Dict"] = self.simulator.pattern_Pack_Dict;
        process_Dict_No_Tensor = {};
        for process_Name in self.simulator.process_Dict.keys():
            process_Dict_No_Tensor[process_Name] = {};
            process_Dict_No_Tensor[process_Name]["Order_List"] = self.simulator.process_Dict[process_Name]["Order_List"];
            process_Dict_No_Tensor[process_Name]["Layer_Control_Dict"] = self.simulator.process_Dict[process_Name]["Layer_Control_Dict"];
            process_Dict_No_Tensor[process_Name]["Connection_Control_Dict"] = self.simulator.process_Dict[process_Name]["Connection_Control_Dict"];
        save_Dict["Process_Dict"] = process_Dict_No_Tensor;
        save_Dict["Learning_Setup_List"] = self.simulator.learning_Setup_List;

        if file_Path[-29:] != ".HNetGUI_Model_Data_for_Macro":
            file_Path += ".HNetGUI_Model_Data_for_Macro";

        with open(file_Path, "wb") as f:
            pickle.dump(save_Dict, f);        

    def Main_UI_about_Button_Clicked(self):        
        self.about_Dialog.show();
        self.main_Dialog.hide();
    def Main_UI_exit_Button_Clicked(self):
        sys.exit();
        
    def Main_UI_Simulator_Changed(self):
        self.main_UI.configVariables_ListWidget.clear();
        self.main_UI.layer_ListWidget.clear();
        self.main_UI.connection_ListWidget.clear();
        self.main_UI.patternPack_ListWidget.clear();
        self.main_UI.process_ListWidget.clear();
        self.main_UI.learningSetup_ListWidget.clear();

        self.main_UI.configVariables_ListWidget.addItem("Momentum: " + str(self.simulator.config_Variables_Dict["Momentum"]));
        self.main_UI.configVariables_ListWidget.addItem("Learning Rate: " + str(self.simulator.config_Variables_Dict["Learning_Rate"]));
        self.main_UI.configVariables_ListWidget.addItem("Decay Rate: " + str(self.simulator.config_Variables_Dict["Decay_Rate"]));
        self.main_UI.configVariables_ListWidget.addItem("Initial Weight SD: " + str(self.simulator.config_Variables_Dict["Initial_Weight_SD"]));
        self.main_UI.configVariables_ListWidget.addItem("Device Mode: " + str(self.simulator.config_Variables_Dict["Device_Mode"]).upper());

        for key in self.simulator.layer_Information_Dict.keys():
            layer_Information = self.simulator.layer_Information_Dict[key];
            self.main_UI.layer_ListWidget.addItem(key + " (" + str(layer_Information["Unit"]) + ")");

        for key in self.simulator.connection_Information_Dict.keys():
            connection_Information = self.simulator.connection_Information_Dict[key];            
            self.main_UI.connection_ListWidget.addItem(key + " (" + str(connection_Information["From_Layer_Name"]) + "→" + str(connection_Information["To_Layer_Name"]) + ")");

        for key in self.simulator.pattern_Pack_Dict.keys():            
            self.main_UI.patternPack_ListWidget.addItem(key);

        for key in self.simulator.process_Dict.keys():            
            self.main_UI.process_ListWidget.addItem(key);

        for learning_Setup in self.simulator.learning_Setup_List:
            self.main_UI.learningSetup_ListWidget.addItem(learning_Setup["Name"]);

    # End Main UI Functions

    # Start Structure Setup Functions    
    def Structure_Setup_UI_summit_Button_Clicked(self):
        if self.structure_Setup_UI.momentum_LineEdit.text() == "":
            self.structure_Setup_UI.momentum_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.decayRate_LineEdit.text() == "":
            self.structure_Setup_UI.decayRate_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.initalWeightSD_LineEdit.text() == "":
            self.structure_Setup_UI.initalWeightSD_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.learningRate_LineEdit.text() == "":
            self.structure_Setup_UI.learningRate_LineEdit.setFocus();
            return;
        
        momentum = float(self.structure_Setup_UI.momentum_LineEdit.text());
        decay_Rate = float(self.structure_Setup_UI.decayRate_LineEdit.text());
        initial_Weight_SD = float(self.structure_Setup_UI.initalWeightSD_LineEdit.text());
        learning_Rate = float(self.structure_Setup_UI.learningRate_LineEdit.text());
        if self.structure_Setup_UI.deviceModeCPU_RadioButton.isChecked():            
            device_Mode = 'cpu'
        elif self.structure_Setup_UI.deviceModeGPU_RadioButton.isChecked():            
            device_Mode = 'gpu'

        self.simulator.Structure_Config_Variable_Setup(momentum=momentum, decay_Rate=decay_Rate, initial_Weight_SD=initial_Weight_SD, learning_Rate=learning_Rate, device_Mode = device_Mode);

        self.Structure_Setup_UI_Structure_Changed();
    
    def Structure_Setup_UI_structureBPMake_Button_Clicked(self):
        if self.structure_Setup_UI.structureBPInputLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureBPInputLayerSize_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.structureBPHiddenLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureBPHiddenLayerSize_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.structureBPOutputLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureBPOutputLayerSize_LineEdit.setFocus();
            return;

        self.simulator.Structure_Layer_Assign(name="Input", unit=int(self.structure_Setup_UI.structureBPInputLayerSize_LineEdit.text()));
        self.simulator.Structure_Layer_Assign(name="Hidden", unit=int(self.structure_Setup_UI.structureBPHiddenLayerSize_LineEdit.text()));
        self.simulator.Structure_Layer_Assign(name="Output", unit=int(self.structure_Setup_UI.structureBPOutputLayerSize_LineEdit.text()));

        self.simulator.Structure_Connection_Assign(name = "IH", from_Layer_Name = "Input", to_Layer_Name = "Hidden");
        self.simulator.Structure_Connection_Assign(name = "HO", from_Layer_Name = "Hidden", to_Layer_Name = "Output");

        self.structure_Setup_UI.structureBPInputLayerSize_LineEdit.setText("");
        self.structure_Setup_UI.structureBPHiddenLayerSize_LineEdit.setText("");
        self.structure_Setup_UI.structureBPOutputLayerSize_LineEdit.setText("");
        
        self.Structure_Setup_UI_Structure_Changed();

    def Structure_Setup_UI_structureBPTTMake_Button_Clicked(self):        
        if self.structure_Setup_UI.structureBPTTInputLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureBPTTInputLayerSize_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.structureBPTTHiddenLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureBPTTHiddenLayerSize_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.structureBPTTOutputLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureBPTTOutputLayerSize_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.structureBPTTTick_LineEdit.text() == "":
            self.structure_Setup_UI.structureBPTTTick_LineEdit.setFocus();
            return;

        max_Tick = int(self.structure_Setup_UI.structureBPTTTick_LineEdit.text());
        
        for tick in range(1, max_Tick + 1):
            self.simulator.Structure_Layer_Assign(name="Input_" + str(tick), unit=int(self.structure_Setup_UI.structureBPTTInputLayerSize_LineEdit.text()));
            self.simulator.Structure_Layer_Assign(name="Hidden_" + str(tick), unit=int(self.structure_Setup_UI.structureBPTTHiddenLayerSize_LineEdit.text()));
        self.simulator.Structure_Layer_Assign(name="Output", unit=int(self.structure_Setup_UI.structureBPTTOutputLayerSize_LineEdit.text()));

        for tick in range(1, max_Tick + 1):
            self.simulator.Structure_Connection_Assign(name = "I" + str(tick) + "H" + str(tick), from_Layer_Name = "Input_" + str(tick), to_Layer_Name = "Hidden_" + str(tick));        
        for tick in range(1, max_Tick):
            self.simulator.Structure_Connection_Assign(name = "H" + str(tick) + "H" + str(tick + 1), from_Layer_Name = "Hidden_" + str(tick), to_Layer_Name = "Hidden_" + str(tick + 1));
        self.simulator.Structure_Connection_Assign(name = "H" + str(max_Tick) + "O", from_Layer_Name = "Hidden_" + str(max_Tick), to_Layer_Name = "Output");

        self.structure_Setup_UI.structureBPTTInputLayerSize_LineEdit.setText("");
        self.structure_Setup_UI.structureBPTTHiddenLayerSize_LineEdit.setText("");
        self.structure_Setup_UI.structureBPTTOutputLayerSize_LineEdit.setText("");
        self.structure_Setup_UI.structureBPTTTick_LineEdit.setText("");
        
        self.Structure_Setup_UI_Structure_Changed();

    def Structure_Setup_UI_structureSRNMake_Button_Clicked(self):        
        if self.structure_Setup_UI.structureSRNInputLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureSRNInputLayerSize_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.structureSRNHiddenLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureSRNHiddenLayerSize_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.structureSRNOutputLayerSize_LineEdit.text() == "":
            self.structure_Setup_UI.structureSRNOutputLayerSize_LineEdit.setFocus();
            return;

        self.simulator.Structure_Layer_Assign(name="Input", unit=int(self.structure_Setup_UI.structureSRNInputLayerSize_LineEdit.text()));
        self.simulator.Structure_Layer_Assign(name="Context", unit=int(self.structure_Setup_UI.structureSRNHiddenLayerSize_LineEdit.text()));
        self.simulator.Structure_Layer_Assign(name="Hidden", unit=int(self.structure_Setup_UI.structureSRNHiddenLayerSize_LineEdit.text()));
        self.simulator.Structure_Layer_Assign(name="Output", unit=int(self.structure_Setup_UI.structureSRNOutputLayerSize_LineEdit.text()));

        self.simulator.Structure_Connection_Assign(name = "IH", from_Layer_Name = "Input", to_Layer_Name = "Hidden");
        self.simulator.Structure_Connection_Assign(name = "CH", from_Layer_Name = "Context", to_Layer_Name = "Hidden");
        self.simulator.Structure_Connection_Assign(name = "HO", from_Layer_Name = "Hidden", to_Layer_Name = "Output");

        self.structure_Setup_UI.structureSRNInputLayerSize_LineEdit.setText("");
        self.structure_Setup_UI.structureSRNHiddenLayerSize_LineEdit.setText("");
        self.structure_Setup_UI.structureSRNOutputLayerSize_LineEdit.setText("");
        
        self.Structure_Setup_UI_Structure_Changed();

    
    def Structure_Setup_UI_layerAdd_Button_Clicked(self):        
        if self.structure_Setup_UI.layerName_LineEdit.text() == "":
            self.structure_Setup_UI.layerName_LineEdit.setFocus();
            return;
        elif self.structure_Setup_UI.layerUnit_LineEdit.text() == "":
            self.structure_Setup_UI.layerUnit_LineEdit.setFocus();
            return;

        name = self.structure_Setup_UI.layerName_LineEdit.text();
        unit = int(self.structure_Setup_UI.layerUnit_LineEdit.text());
        self.simulator.Structure_Layer_Assign(name=name, unit=unit);

        self.structure_Setup_UI.layerName_LineEdit.setText("");
        self.structure_Setup_UI.layerUnit_LineEdit.setText("");
        
        self.Structure_Setup_UI_Structure_Changed();
        
    def Structure_Setup_UI_layerDelete_Button_Clicked(self):
        if self.structure_Setup_UI.layer_ListWidget.currentRow() < 0:
            return;

        selected_Layer_Name = self.structure_Setup_UI.layer_ListWidget.currentItem().text().split(' ')[0];
        self.simulator.Structure_Layer_Delete(selected_Layer_Name);

        self.Structure_Setup_UI_Structure_Changed();
        
    def Structure_Setup_UI_connectionAdd_Button_Clicked(self):
        if self.structure_Setup_UI.connectionName_LineEdit.text() == "":
            self.structure_Setup_UI.connectionName_LineEdit.setFocus();
            return;

        name = self.structure_Setup_UI.connectionName_LineEdit.text();
        from_Layer_Name = self.structure_Setup_UI.connectionFrom_ComboBox.currentText(); 
        to_Layer_Name = self.structure_Setup_UI.connectionTo_ComboBox.currentText();

        self.simulator.Structure_Connection_Assign(name = name, from_Layer_Name = from_Layer_Name, to_Layer_Name = to_Layer_Name)
        
        self.structure_Setup_UI.connectionName_LineEdit.setText("");
        self.structure_Setup_UI.connectionFrom_ComboBox.setCurrentIndex(0);
        self.structure_Setup_UI.connectionTo_ComboBox.setCurrentIndex(0);

        self.Structure_Setup_UI_Structure_Changed();

    def Structure_Setup_UI_connectionDelete_Button_Clicked(self):
        if self.structure_Setup_UI.connection_ListWidget.currentRow() < 0:
            return;

        selected_Connection_Name = self.structure_Setup_UI.connection_ListWidget.currentItem().text().split(' ')[0];
        self.simulator.Structure_Connection_Delete(selected_Connection_Name);

        self.Structure_Setup_UI_Structure_Changed();
    
    def Structure_Setup_UI_structureSave_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getSaveFileName(filter="Structure File for HNet (*.HNet_Structure)")[0]
        if file_Path != "":
            self.simulator.Structure_Save(file_Path);

    def Structure_Setup_UI_structureLoad_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getOpenFileName(filter="Structure File for HNet (*.HNet_Structure)")[0];
        if file_Path != "":
            self.simulator.Structure_Load(file_Path);

        self.Structure_Setup_UI_Structure_Changed();
    
    def Structure_Setup_UI_exit_Button_Clicked(self):
        self.structure_Setup_Dialog.hide();
        if len(self.simulator.connection_Information_Dict) > 0:
            self.main_UI.structureLock_Button.setEnabled(True);
        else:
            self.main_UI.structureLock_Button.setEnabled(False);
        self.Main_UI_Simulator_Changed();
        self.main_Dialog.exec_();

    def Structure_Setup_UI_Structure_Changed(self):
        self.structure_Setup_UI.momentum_LineEdit.setText(str(self.simulator.config_Variables_Dict["Momentum"]));
        self.structure_Setup_UI.decayRate_LineEdit.setText(str(self.simulator.config_Variables_Dict["Decay_Rate"]));
        self.structure_Setup_UI.initalWeightSD_LineEdit.setText(str(self.simulator.config_Variables_Dict["Initial_Weight_SD"]));
        self.structure_Setup_UI.learningRate_LineEdit.setText(str(self.simulator.config_Variables_Dict["Learning_Rate"]));
        if self.simulator.config_Variables_Dict["Device_Mode"] == 'cpu':
            self.structure_Setup_UI.deviceModeCPU_RadioButton.setChecked(True);
        elif self.simulator.config_Variables_Dict["Device_Mode"] == 'gpu':
            self.structure_Setup_UI.deviceModeGPU_RadioButton.setChecked(True);

        self.structure_Setup_UI.layer_ListWidget.clear();

        for key in self.simulator.layer_Information_Dict.keys():
            layer_Information = self.simulator.layer_Information_Dict[key];            
            self.structure_Setup_UI.layer_ListWidget.addItem(key + " (" + str(layer_Information["Unit"]) + ")");

        self.structure_Setup_UI.connectionFrom_ComboBox.clear();        
        self.structure_Setup_UI.connectionTo_ComboBox.clear();
        for key in self.simulator.layer_Information_Dict.keys():
            self.structure_Setup_UI.connectionFrom_ComboBox.addItem(key);
            self.structure_Setup_UI.connectionTo_ComboBox.addItem(key);

        self.structure_Setup_UI.connection_ListWidget.clear();

        for key in self.simulator.connection_Information_Dict.keys():
            connection_Information = self.simulator.connection_Information_Dict[key];
            
            connection_Text_List = [];
            connection_Text_List.append(key + " (" + str(connection_Information["From_Layer_Name"]) + "→" + str(connection_Information["To_Layer_Name"]) + ")");            

            self.structure_Setup_UI.connection_ListWidget.addItem("".join(connection_Text_List));

        if len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU']) < 1:
            self.structure_Setup_UI.deviceModeGPU_RadioButton.setEnabled(False);
        else:
            self.structure_Setup_UI.deviceModeGPU_RadioButton.setEnabled(True);
        
    # End Structure Setup Functions
    
    # Start Pattern Setup Functions
    def Pattern_Setup_UI_broswer_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getOpenFileName(filter="Pattern Data TXT File for HNet (*.txt)")[0];
        if file_Path != "":
            self.pattern_Setup_UI.filePath_LineEdit.setText(file_Path);

    def Pattern_Setup_UI_insert_Button_Clicked(self):
        if self.pattern_Setup_UI.filePath_LineEdit.text() == "":
            self.pattern_Setup_UI.broswer_Button.setFocus();            
            return;
        elif self.pattern_Setup_UI.packName_LineEdit.text() == "": 
            self.pattern_Setup_UI.packName_LineEdit.setFocus();
            return;

        self.simulator.Pattern_Pack_Load(self.pattern_Setup_UI.packName_LineEdit.text(), self.pattern_Setup_UI.filePath_LineEdit.text())
        self.Pattern_Setup_UI_Pattern_Pack_Changed();

    def Pattern_Setup_UI_delete_Button_Clicked(self):
        if self.pattern_Setup_UI.patternPack_ListWidget.currentRow() < 0:
            return;

        selected_Pattern_Pack_Name = self.pattern_Setup_UI.patternPack_ListWidget.currentItem().text();
        self.simulator.Pattern_Pack_Delete(selected_Pattern_Pack_Name);
        self.Pattern_Setup_UI_Pattern_Pack_Changed();

    def Pattern_Setup_UI_patternPack_ListWidget_Current_Item_Changed(self):        
        self.pattern_Setup_UI.patternPackInformation_TextEdit.setText("");
        
        if self.pattern_Setup_UI.patternPack_ListWidget.currentRow() < 0:
            return

        selected_Pattern_Pack = self.simulator.pattern_Pack_Dict[self.pattern_Setup_UI.patternPack_ListWidget.currentItem().text()];
        pack_Information_Text_List = [];
        pack_Information_Text_List.append("Count :" + str(selected_Pattern_Pack["Count"]));

        for key in selected_Pattern_Pack.keys():
            if key in ["Name", "Probability", "Cycle", "Count"]:
                continue;
            pack_Information_Text_List.append(key + ": " + str(selected_Pattern_Pack[key].shape));

        self.pattern_Setup_UI.patternPackInformation_TextEdit.setText("\n".join(pack_Information_Text_List));

    def Pattern_Setup_UI_exit_Button_Clicked(self):
        self.pattern_Setup_Dialog.hide();
        self.Main_UI_Simulator_Changed();
        self.main_Dialog.exec_();
    
    def Pattern_Setup_UI_Pattern_Pack_Changed(self):
        self.pattern_Setup_UI.filePath_LineEdit.setText("");
        self.pattern_Setup_UI.packName_LineEdit.setText("");
        self.pattern_Setup_UI.patternPack_ListWidget.clear();        

        for key in self.simulator.pattern_Pack_Dict.keys():            
            self.pattern_Setup_UI.patternPack_ListWidget.addItem(key);
    # End Pattern Setup Functions
    
    # Start Process Setup Functions
    def Process_Setup_UI_processMaking_Button_Clicked(self):
        if self.process_Setup_UI.processName_LineEdit.text() == "":
            self.process_Setup_UI.processName_LineEdit.setFocus();            
            return;
            
        self.current_Process_Order_List = [];
        self.current_Process_Layer_Control_Dict = {};
        self.current_Process_Connection_Control_Dict = {};
        for layer in self.simulator.layer_Information_Dict.keys():
            self.current_Process_Layer_Control_Dict[layer] = (Damage_Type.On, None);
        for connection in self.simulator.connection_Information_Dict.keys():
            self.current_Process_Connection_Control_Dict[connection] = (Damage_Type.On, None);

        self.current_Process_Linear_Forward_List = [];
        self.current_Process_Linear_Backward_List = [];
        self.current_Process_Custom_Bias_Equalization_Layer_List = [];
        self.current_Process_Custom_Weight_Equalization_Connection_List = [];

        self.Process_Setup_UI_Order_and_Control_Changed();
        self.Process_Setup_UI_Making_Mode_Widget_Enable();
        
    def Process_Setup_UI_processModify_Button_Clicked(self):
        if self.process_Setup_UI.process_ListWidget.currentRow() < 0:
            return;

        selected_Process_Name = self.process_Setup_UI.process_ListWidget.currentItem().text();
        self.process_Setup_UI.processName_LineEdit.setText(selected_Process_Name);
        self.current_Process_Order_List = self.simulator.process_Dict[selected_Process_Name]["Order_List"];
        self.current_Process_Layer_Control_Dict = self.simulator.process_Dict[selected_Process_Name]["Layer_Control_Dict"];
        self.current_Process_Connection_Control_Dict = self.simulator.process_Dict[selected_Process_Name]["Connection_Control_Dict"];

        self.current_Process_Linear_Forward_List = [];
        self.current_Process_Linear_Backward_List = [];
        self.current_Process_Custom_Bias_Equalization_Layer_List = [];
        self.current_Process_Custom_Weight_Equalization_Connection_List = [];

        self.Process_Setup_UI_Order_and_Control_Changed();
        self.Process_Setup_UI_Making_Mode_Widget_Enable();

    def Process_Setup_UI_processEnd_Button_Clicked(self):
        if len(self.current_Process_Order_List) < 1:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no order.");
            return;
        elif not self.current_Process_Order_List[-1][0] == Order_Code.End_and_Initialize:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "Last order should be the 'End & Initialize'.");
            return;

        self.simulator.Process_Assign(self.process_Setup_UI.processName_LineEdit.text(), self.current_Process_Order_List, self.current_Process_Layer_Control_Dict, self.current_Process_Connection_Control_Dict);
        
        self.current_Process_Order_List = [];
        self.current_Process_Layer_Control_Dict = {};
        self.current_Process_Connection_Control_Dict = {};

        self.current_Process_Linear_Forward_List = [];
        self.current_Process_Linear_Backward_List = [];

        self.Process_Setup_UI_Process_Changed();
        self.Process_Setup_UI_Order_and_Control_Changed();
        self.Process_Setup_UI_Viewing_Mode_Widget_Enable();

        self.process_Setup_UI.process_ListWidget.setCurrentRow(-1);

    def Process_Setup_UI_process_ListWidget_Current_Item_Changed(self):
        if self.process_Setup_UI.process_ListWidget.currentRow() < 0:
            self.current_Process_Order_List = [];
            self.current_Process_Layer_Control_Dict = {};
            self.current_Process_Connection_Control_Dict = {};
        else:
            selected_Process_Name = self.process_Setup_UI.process_ListWidget.currentItem().text();
            self.current_Process_Order_List = self.simulator.process_Dict[selected_Process_Name]["Order_List"];
            self.current_Process_Layer_Control_Dict = self.simulator.process_Dict[selected_Process_Name]["Layer_Control_Dict"];
            self.current_Process_Connection_Control_Dict = self.simulator.process_Dict[selected_Process_Name]["Connection_Control_Dict"];

        self.Process_Setup_UI_Order_and_Control_Changed();
    
    def Process_Setup_UI_processDelete_Button_Clicked(self):
        if self.process_Setup_UI.process_ListWidget.currentRow() < 0:
            return;

        selected_Process_Name = self.process_Setup_UI.process_ListWidget.currentItem().text();
        self.simulator.Process_Delete(selected_Process_Name);
        self.Process_Setup_UI_Process_Changed();

    def Process_Setup_UI_layerOn_Button_Clicked(self):
        if self.process_Setup_UI.layerControl_ListWidget.currentRow() < 0:
            return;

        selected_Layer_Name = self.process_Setup_UI.layerControl_ListWidget.currentItem().text().split(" ")[0];
        self.current_Process_Layer_Control_Dict[selected_Layer_Name] = (Damage_Type.On, None);

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_layerOff_Button_Clicked(self):
        if self.process_Setup_UI.layerControl_ListWidget.currentRow() < 0:
            return;

        selected_Layer_Name = self.process_Setup_UI.layerControl_ListWidget.currentItem().text().split(" ")[0];
        self.current_Process_Layer_Control_Dict[selected_Layer_Name] = (Damage_Type.Off, None);

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_layerDamage_Button_Clicked(self):
        if self.process_Setup_UI.layerControl_ListWidget.currentRow() < 0:
            return;

        selected_Layer_Name = self.process_Setup_UI.layerControl_ListWidget.currentItem().text().split(" ")[0];
        self.current_Process_Layer_Control_Dict[selected_Layer_Name] = (Damage_Type.Damaged, float(self.process_Setup_UI.layerDamageSD_LineEdit.text()));

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_connectionOn_Button_Clicked(self):
        if self.process_Setup_UI.connectionControl_ListWidget.currentRow() < 0:
            return;

        selected_Connection_Name = self.process_Setup_UI.connectionControl_ListWidget.currentItem().text().split(" ")[0];
        self.current_Process_Connection_Control_Dict[selected_Connection_Name] = (Damage_Type.On, None);

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_connectionOff_Button_Clicked(self):
        if self.process_Setup_UI.connectionControl_ListWidget.currentRow() < 0:
            return;

        selected_Connection_Name = self.process_Setup_UI.connectionControl_ListWidget.currentItem().text().split(" ")[0];
        self.current_Process_Connection_Control_Dict[selected_Connection_Name] = (Damage_Type.Off, None);

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_connectionDamage_Button_Clicked(self):
        if self.process_Setup_UI.connectionControl_ListWidget.currentRow() < 0:
            return;

        selected_Connection_Name = self.process_Setup_UI.connectionControl_ListWidget.currentItem().text().split(" ")[0];
        self.current_Process_Connection_Control_Dict[selected_Connection_Name] = (Damage_Type.Damaged, float(self.process_Setup_UI.connectionDamageSD_LineEdit.text()));

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_orderDelete_Button_Clicked(self):
        selected_Order_Index = self.process_Setup_UI.order_ListWidget.currentRow();
        if selected_Order_Index < 0:
            return;

        deleted_Order = self.current_Process_Order_List[selected_Order_Index]; 

        del self.current_Process_Order_List[selected_Order_Index];

        for order_Index in range(selected_Order_Index, len(self.current_Process_Order_List)):
            if not self.Process_Setup_UI_Order_Consistency_Check(order_Index):
                self.current_Process_Order_List.insert(selected_Order_Index, deleted_Order)    #Undo                
                QtWidgets.QMessageBox.warning(None, 'Warning!', "This order deleting impairs the consistency of orders.");
                break;

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_orderUp_Button_Clicked(self):
        selected_Order_Index = self.process_Setup_UI.order_ListWidget.currentRow();        
        if not selected_Order_Index > 0:
            return;
            
        self.current_Process_Order_List[selected_Order_Index], self.current_Process_Order_List[selected_Order_Index - 1] = self.current_Process_Order_List[selected_Order_Index - 1], self.current_Process_Order_List[selected_Order_Index];

        if not self.Process_Setup_UI_Order_Consistency_Check(selected_Order_Index - 1) or not self.Process_Setup_UI_Order_Consistency_Check(selected_Order_Index):
            self.current_Process_Order_List[selected_Order_Index], self.current_Process_Order_List[selected_Order_Index - 1] = self.current_Process_Order_List[selected_Order_Index - 1], self.current_Process_Order_List[selected_Order_Index];    #Undo
            self.process_Setup_UI.order_ListWidget.setCurrentRow(selected_Order_Index);
            QtWidgets.QMessageBox.warning(None, 'Warning!', "This order move impairs the consistency of orders.");            

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_orderDown_Button_Clicked(self):
        selected_Order_Index = self.process_Setup_UI.order_ListWidget.currentRow();
        if not selected_Order_Index < len(self.current_Process_Order_List) - 1:
            return;

        self.current_Process_Order_List[selected_Order_Index], self.current_Process_Order_List[selected_Order_Index + 1] = self.current_Process_Order_List[selected_Order_Index + 1], self.current_Process_Order_List[selected_Order_Index];
        self.process_Setup_UI.order_ListWidget.setCurrentRow(selected_Order_Index + 1);
    
        if not self.Process_Setup_UI_Order_Consistency_Check(selected_Order_Index) or not self.Process_Setup_UI_Order_Consistency_Check(selected_Order_Index + 1):
            self.current_Process_Order_List[selected_Order_Index], self.current_Process_Order_List[selected_Order_Index + 1] = self.current_Process_Order_List[selected_Order_Index + 1], self.current_Process_Order_List[selected_Order_Index];    #Undo;
            self.process_Setup_UI.order_ListWidget.setCurrentRow(selected_Order_Index);           
            QtWidgets.QMessageBox.warning(None, 'Warning!', "This order move impairs the consistency of orders.");

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_bpInputLayer_ComboBox_Current_Index_Changed(self):
        self.process_Setup_UI.bpHiddenLayer_ComboBox.clear();
        self.process_Setup_UI.bpOutputLayer_ComboBox.clear();

        extract_Information = self.simulator.Extract_Connection_List(self.process_Setup_UI.bpInputLayer_ComboBox.currentText());
        for connection, layer in extract_Information:
            self.process_Setup_UI.bpHiddenLayer_ComboBox.addItem(layer);

    def Process_Setup_UI_bpHiddenLayer_ComboBox_Current_Index_Changed(self):        
        self.process_Setup_UI.bpOutputLayer_ComboBox.clear();

        extract_Information = self.simulator.Extract_Connection_List(self.process_Setup_UI.bpHiddenLayer_ComboBox.currentText());
        for connection, layer in extract_Information:
            self.process_Setup_UI.bpOutputLayer_ComboBox.addItem(layer);

    def Process_Setup_UI_bpTrainingApply_Button_Clicked(self):
        if self.process_Setup_UI.bpInputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpInputLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.bpHiddenLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpHiddenLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.bpOutputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpOutputLayer_ComboBox.setFocus();
            return;

        input_Layer = self.process_Setup_UI.bpInputLayer_ComboBox.currentText();
        hidden_Layer = self.process_Setup_UI.bpHiddenLayer_ComboBox.currentText();
        output_Layer = self.process_Setup_UI.bpOutputLayer_ComboBox.currentText();
        
        input_to_Hidden_Connection = self.simulator.Extract_Connection(input_Layer, hidden_Layer);
        hidden_to_Output_Connection = self.simulator.Extract_Connection(hidden_Layer, output_Layer);
        
        self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [input_Layer], None, None));        
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [input_Layer, hidden_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer, output_Layer], None, None));
        if self.process_Setup_UI.bpErrorSigmoid_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [output_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Sigmoid, [output_Layer], None, None));
        elif self.process_Setup_UI.bpErrorSoftmax_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [output_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Softmax, [output_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Error_Send, [output_Layer, hidden_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [hidden_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [input_to_Hidden_Connection], None));
        self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [hidden_to_Output_Connection], None));
        self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [hidden_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [output_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.End_and_Initialize, None, None, None));

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_bpTestApply_Button_Clicked(self):
        if self.process_Setup_UI.bpInputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpInputLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.bpHiddenLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpHiddenLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.bpOutputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpOutputLayer_ComboBox.setFocus();
            return;

        input_Layer = self.process_Setup_UI.bpInputLayer_ComboBox.currentText();
        hidden_Layer = self.process_Setup_UI.bpHiddenLayer_ComboBox.currentText();
        output_Layer = self.process_Setup_UI.bpOutputLayer_ComboBox.currentText();
        
        input_to_Hidden_Connection = self.simulator.Extract_Connection(input_Layer, hidden_Layer);
        hidden_to_Output_Connection = self.simulator.Extract_Connection(hidden_Layer, output_Layer);
        
        self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [input_Layer], None, None));        
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [input_Layer, hidden_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer, output_Layer], None, None));
        if self.process_Setup_UI.bpErrorSigmoid_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [output_Layer], None, None));            
        elif self.process_Setup_UI.bpErrorSoftmax_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [output_Layer], None, None));            
        self.current_Process_Order_List.append((Order_Code.Activation_Extract, [output_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.End_and_Initialize, None, None, None));

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_bpttTrainingApply_Button_Clicked(self):        
        if self.process_Setup_UI.bpttInputLayer_LineEdit.text() == "":
            self.process_Setup_UI.bpttInputLayer_LineEdit.setFocus();
            return;
        elif self.process_Setup_UI.bpttHiddenLayer_LineEdit.text() == "":
            self.process_Setup_UI.bpttHiddenLayer_LineEdit.setFocus();
            return;
        elif self.process_Setup_UI.bpttTick_LineEdit.text() == "":
            self.process_Setup_UI.bpttTick_LineEdit.setFocus();
            return;
        elif self.process_Setup_UI.bpttOutputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpttOutputLayer_ComboBox.setFocus();
            return;

        input_Layer_Regular_Name = self.process_Setup_UI.bpttInputLayer_LineEdit.text();
        hidden_Layer_Regular_Name = self.process_Setup_UI.bpttHiddenLayer_LineEdit.text();
        max_Tick = int(self.process_Setup_UI.bpttTick_LineEdit.text());
        
        input_Layer_List = [];
        hidden_Layer_List = [];
        for tick in range(1, max_Tick + 1):
            input_Layer_List.append(input_Layer_Regular_Name + str(tick));
            hidden_Layer_List.append(hidden_Layer_Regular_Name + str(tick));                
        if not all([x in self.simulator.layer_Information_Dict.keys() for x in input_Layer_List]) or not all([x in self.simulator.layer_Information_Dict.keys() for x in hidden_Layer_List]):
             QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no layer. The prefix of layer name or tick maybe wrong");
             return;             
        output_Layer = self.process_Setup_UI.bpttOutputLayer_ComboBox.currentText();
        
        input_to_Hidden_Connection_List = [];
        for index in range(max_Tick):
            input_to_Hidden_Connection_List.append(self.simulator.Extract_Connection(input_Layer_List[index], hidden_Layer_List[index]));
        hidden_to_Hidden_Connection_List = [];
        for index in range(max_Tick - 1):
            hidden_to_Hidden_Connection_List.append(self.simulator.Extract_Connection(hidden_Layer_List[index], hidden_Layer_List[index + 1]));        
        hidden_to_Output_Connection = self.simulator.Extract_Connection(hidden_Layer_List[-1], output_Layer);
        if not hidden_to_Output_Connection:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no connection between the last hidden and output layers.");
            return;

        for index in range(len(input_Layer_List)):
            self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [input_Layer_List[index]], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [input_Layer_List[index], hidden_Layer_List[index]], None, None));
        for index in range(len(hidden_Layer_List) - 1):
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer_List[index]], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer_List[index], hidden_Layer_List[index + 1]], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer_List[-1]], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer_List[-1], output_Layer], None, None));                        
        if self.process_Setup_UI.bpErrorSigmoid_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [output_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Sigmoid, [output_Layer], None, None));
        elif self.process_Setup_UI.bpErrorSoftmax_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [output_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Softmax, [output_Layer], None, None));        
        self.current_Process_Order_List.append((Order_Code.Error_Send, [output_Layer, hidden_Layer_List[-1]], None, None));
        for index in reversed(range(1, len(hidden_Layer_List))):
            self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [hidden_Layer_List[index]], None, None));
            self.current_Process_Order_List.append((Order_Code.Error_Send, [hidden_Layer_List[index], hidden_Layer_List[index - 1]], None, None));
        self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [hidden_Layer_List[0]], None, None));        
        for input_to_Hidden_Connection in input_to_Hidden_Connection_List:
            self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [input_to_Hidden_Connection], None));
        for hidden_to_Hidden_Connection in hidden_to_Hidden_Connection_List:
            self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [hidden_to_Hidden_Connection], None));
        for hidden_Layer in hidden_Layer_List:
            self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [hidden_Layer], None, None));    
        self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [output_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Weight_Equalization, None, input_to_Hidden_Connection_List, None));
        self.current_Process_Order_List.append((Order_Code.Weight_Equalization, None, hidden_to_Hidden_Connection_List, None));
        self.current_Process_Order_List.append((Order_Code.Bias_Equalization, input_Layer_List, None, None));
        self.current_Process_Order_List.append((Order_Code.Bias_Equalization, hidden_Layer_List, None, None));
        self.current_Process_Order_List.append((Order_Code.End_and_Initialize, None, None, None));

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_bpttTestApply_Button_Clicked(self):
        if self.process_Setup_UI.bpttInputLayer_LineEdit.text() == "":
            self.process_Setup_UI.bpttInputLayer_LineEdit.setFocus();
            return;
        elif self.process_Setup_UI.bpttHiddenLayer_LineEdit.text() == "":
            self.process_Setup_UI.bpttHiddenLayer_LineEdit.setFocus();
            return;
        elif self.process_Setup_UI.bpttTick_LineEdit.text() == "":
            self.process_Setup_UI.bpttTick_LineEdit.setFocus();
            return;
        elif self.process_Setup_UI.bpttOutputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.bpttOutputLayer_ComboBox.setFocus();
            return;
            
        input_Layer_Regular_Name = self.process_Setup_UI.bpttInputLayer_LineEdit.text();
        hidden_Layer_Regular_Name = self.process_Setup_UI.bpttHiddenLayer_LineEdit.text();
        max_Tick = int(self.process_Setup_UI.bpttTick_LineEdit.text());
        
        input_Layer_List = [];
        hidden_Layer_List = [];
        for tick in range(1, max_Tick + 1):
            input_Layer_List.append(input_Layer_Regular_Name + str(tick));
            hidden_Layer_List.append(hidden_Layer_Regular_Name + str(tick));                
        if not all([x in self.simulator.layer_Information_Dict.keys() for x in input_Layer_List]) or not all([x in self.simulator.layer_Information_Dict.keys() for x in hidden_Layer_List]):
             QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no layer. The prefix of layer name or tick maybe wrong");
             return;
        output_Layer = self.process_Setup_UI.bpttOutputLayer_ComboBox.currentText();
        
        input_to_Hidden_Connection_List = [];
        for index in range(max_Tick):
            input_to_Hidden_Connection_List.append(self.simulator.Extract_Connection(input_Layer_List[index], hidden_Layer_List[index]));
        hidden_to_Hidden_Connection_List = [];
        for index in range(max_Tick - 1):
            hidden_to_Hidden_Connection_List.append(self.simulator.Extract_Connection(hidden_Layer_List[index], hidden_Layer_List[index + 1]));        
        hidden_to_Output_Connection = self.simulator.Extract_Connection(hidden_Layer_List[-1], output_Layer);
        if not hidden_to_Output_Connection:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no connection between the last hidden and output layers.");
            return;

        for index in range(len(input_Layer_List)):
            self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [input_Layer_List[index]], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [input_Layer_List[index], hidden_Layer_List[index]], None, None));
        for index in range(len(hidden_Layer_List) - 1):
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer_List[index]], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer_List[index], hidden_Layer_List[index + 1]], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer_List[-1]], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer_List[-1], output_Layer], None, None));                        
        if self.process_Setup_UI.bpErrorSigmoid_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [output_Layer], None, None));        
        elif self.process_Setup_UI.bpErrorSoftmax_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [output_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Extract, [output_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.End_and_Initialize, None, None, None));

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_srnInputLayer_ComboBox_Current_Index_Changed(self):
        self.process_Setup_UI.srnHiddenLayer_ComboBox.clear();
        self.process_Setup_UI.srnOutputLayer_ComboBox.clear();

        if self.process_Setup_UI.srnInputLayer_ComboBox.currentIndex() > -1 and self.process_Setup_UI.srnContextLayer_ComboBox.currentIndex() > -1:
            extract_Information_from_Input = self.simulator.Extract_Connection_List(self.process_Setup_UI.srnInputLayer_ComboBox.currentText());
            extract_Information_from_Context = self.simulator.Extract_Connection_List(self.process_Setup_UI.srnContextLayer_ComboBox.currentText());
            hidden_Candidate_from_Input = [x[1] for x in extract_Information_from_Input];
            hidden_Candidate_from_Context = [x[1] for x in extract_Information_from_Context];
            layer_Candidate_List = list(set(hidden_Candidate_from_Input) & set(hidden_Candidate_from_Context));
            for layer in layer_Candidate_List:
                if self.simulator.layer_Information_Dict[self.process_Setup_UI.srnContextLayer_ComboBox.currentText()]["Unit"] == self.simulator.layer_Information_Dict[layer]["Unit"]:  #Conext layer's unit amount has to be same to hidden layer's 
                    self.process_Setup_UI.srnHiddenLayer_ComboBox.addItem(layer);

    def Process_Setup_UI_srnContextLayer_ComboBox_Current_Index_Changed(self):
        self.Process_Setup_UI_srnInputLayer_ComboBox_Current_Index_Changed();

    def Process_Setup_UI_srnHiddenLayer_ComboBox_Current_Index_Changed(self):
        self.process_Setup_UI.srnOutputLayer_ComboBox.clear();

        extract_Information = self.simulator.Extract_Connection_List(self.process_Setup_UI.srnHiddenLayer_ComboBox.currentText());
        for connection, layer in extract_Information:
            self.process_Setup_UI.srnOutputLayer_ComboBox.addItem(layer);

    def Process_Setup_UI_srnTrainingApply_Button_Clicked(self):
        if self.process_Setup_UI.srnInputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnInputLayer_ComboBox.setFocus();
            return;
        if self.process_Setup_UI.srnContextLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnContextLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.srnHiddenLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnHiddenLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.srnOutputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnOutputLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.srnMaxCycle_LineEdit.text() == "":
            self.process_Setup_UI.srnMaxCycle_LineEdit.setFocus();
            return;

        input_Layer = self.process_Setup_UI.srnInputLayer_ComboBox.currentText();
        context_Layer = self.process_Setup_UI.srnContextLayer_ComboBox.currentText();
        hidden_Layer = self.process_Setup_UI.srnHiddenLayer_ComboBox.currentText();
        output_Layer = self.process_Setup_UI.srnOutputLayer_ComboBox.currentText();
        max_Cycle = int(self.process_Setup_UI.srnMaxCycle_LineEdit.text());
        
        input_to_Hidden_Connection = self.simulator.Extract_Connection(input_Layer, hidden_Layer);
        context_to_Hidden_Connection = self.simulator.Extract_Connection(context_Layer, hidden_Layer);
        hidden_to_Output_Connection = self.simulator.Extract_Connection(hidden_Layer, output_Layer);
        
        for cycle in range(max_Cycle):
            self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [input_Layer], None, None));        
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [input_Layer, hidden_Layer], None, None));
            if cycle != 0:
                self.current_Process_Order_List.append((Order_Code.Activation_Send, [context_Layer, hidden_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer, output_Layer], None, None));
            if self.process_Setup_UI.srnErrorSigmoid_RadioButton.isChecked():
                self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [output_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Sigmoid, [output_Layer], None, None));
            elif self.process_Setup_UI.srnErrorSoftmax_RadioButton.isChecked():
                self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [output_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Softmax, [output_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Error_Send, [output_Layer, hidden_Layer], None, None));    
            self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [hidden_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [input_to_Hidden_Connection], None));
            if cycle != 0:
                self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [context_to_Hidden_Connection], None));        
            self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [hidden_to_Output_Connection], None));
            self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [hidden_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [output_Layer], None, None));
            if cycle < max_Cycle - 1:
                self.current_Process_Order_List.append((Order_Code.Layer_Duplication, [hidden_Layer, context_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Layer_Initialize, [input_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Layer_Initialize, [hidden_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Layer_Initialize, [output_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Cycle_Maker, None, None, None));
        self.current_Process_Order_List.append((Order_Code.End_and_Initialize, None, None, None));

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_srnTestApply_Button_Clicked(self):
        if self.process_Setup_UI.srnInputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnInputLayer_ComboBox.setFocus();
            return;
        if self.process_Setup_UI.srnContextLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnContextLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.srnHiddenLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnHiddenLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.srnOutputLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.srnOutputLayer_ComboBox.setFocus();
            return;
        elif self.process_Setup_UI.srnMaxCycle_LineEdit.text() == "":
            self.process_Setup_UI.srnMaxCycle_LineEdit.setFocus();
            return;

        input_Layer = self.process_Setup_UI.srnInputLayer_ComboBox.currentText();
        context_Layer = self.process_Setup_UI.srnContextLayer_ComboBox.currentText();
        hidden_Layer = self.process_Setup_UI.srnHiddenLayer_ComboBox.currentText();
        output_Layer = self.process_Setup_UI.srnOutputLayer_ComboBox.currentText();
        max_Cycle = int(self.process_Setup_UI.srnMaxCycle_LineEdit.text());
        
        input_to_Hidden_Connection = self.simulator.Extract_Connection(input_Layer, hidden_Layer);
        context_to_Hidden_Connection = self.simulator.Extract_Connection(context_Layer, hidden_Layer);
        hidden_to_Output_Connection = self.simulator.Extract_Connection(hidden_Layer, output_Layer);
        
        for cycle in range(max_Cycle):
            self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [input_Layer], None, None));        
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [input_Layer, hidden_Layer], None, None));
            if cycle != 0:
                self.current_Process_Order_List.append((Order_Code.Activation_Send, [context_Layer, hidden_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [hidden_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [hidden_Layer, output_Layer], None, None));
            if self.process_Setup_UI.srnErrorSigmoid_RadioButton.isChecked():
                self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [output_Layer], None, None));
            elif self.process_Setup_UI.srnErrorSoftmax_RadioButton.isChecked():
                self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [output_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Extract, [output_Layer], None, None));
            if cycle < max_Cycle - 1:
                self.current_Process_Order_List.append((Order_Code.Layer_Duplication, [hidden_Layer, context_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Layer_Initialize, [input_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Layer_Initialize, [hidden_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Layer_Initialize, [output_Layer], None, None));
                self.current_Process_Order_List.append((Order_Code.Cycle_Maker, None, None, None));            
        self.current_Process_Order_List.append((Order_Code.End_and_Initialize, None, None, None));

        self.Process_Setup_UI_Order_and_Control_Changed();
    
    def Process_Setup_UI_linearForwardAdd_Button_Clicked(self):
        if self.process_Setup_UI.linearForwardLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.linearForwardLayer_ComboBox.setFocus();
            return;

        self.current_Process_Linear_Forward_List.append(self.process_Setup_UI.linearForwardLayer_ComboBox.currentText());
        
        self.Process_Setup_UI_linearForwardLayer_Changed();

    def Process_Setup_UI_linearForwardDelete_Button_Clicked(self):
        if len(self.current_Process_Linear_Forward_List) < 1:
            return;

        del self.current_Process_Linear_Forward_List[-1];

        self.Process_Setup_UI_linearForwardLayer_Changed();
    
    def Process_Setup_UI_linearForwardApply_Button_Clicked(self):
        if len(self.current_Process_Linear_Forward_List) < 2:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "At least, there are two layers in the list.");
            return;

        first_Layer = self.current_Process_Linear_Forward_List[0];
        last_Layer = self.current_Process_Linear_Forward_List[-1];

        if self.process_Setup_UI.linearForwardFirstLayerTypeInput_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [first_Layer], None, None));            
        elif self.process_Setup_UI.linearForwardFirstLayerTypeHidden_RadioButton.isChecked():
            if not self.Process_Setup_UI_Layer_Activated_Check(first_Layer) and self.Process_Setup_UI_Layer_Activation_Stroaged_Check(first_Layer):
                self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [first_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [first_Layer, self.current_Process_Linear_Forward_List[1]], None, None));
        for layer_Index in range(1, len(self.current_Process_Linear_Forward_List) - 1):
            layer = self.current_Process_Linear_Forward_List[layer_Index];
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Activation_Send, [layer, self.current_Process_Linear_Forward_List[layer_Index + 1]], None, None));
        if self.process_Setup_UI.linearForwardLastLayerTypeSigmoid_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [last_Layer], None, None));
        elif self.process_Setup_UI.linearForwardLastLayerTypeSoftmax_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [last_Layer], None, None));
        if self.process_Setup_UI.linearForwardActivationExtractforTest_CheckBox.isChecked():
            self.current_Process_Order_List.append((Order_Code.Activation_Extract, [last_Layer], None, None));

        self.current_Process_Linear_Forward_List = [];    
        
        self.Process_Setup_UI_Order_and_Control_Changed();
        self.Process_Setup_UI_linearForwardLayer_Changed();

    def Process_Setup_UI_linearBackwardAdd_Button_Clicked(self):
        if self.process_Setup_UI.linearBackwardLayer_ComboBox.currentText() == "":
            self.process_Setup_UI.linearBackwardLayer_ComboBox.setFocus();
            return;

        self.current_Process_Linear_Backward_List.append(self.process_Setup_UI.linearBackwardLayer_ComboBox.currentText());
        
        self.Process_Setup_UI_linearBackwardLayer_Changed();

    def Process_Setup_UI_linearBackwardDelete_Button_Clicked(self):
        if len(self.current_Process_Linear_Backward_List) < 1:
            return;

        del self.current_Process_Linear_Backward_List[-1];

        self.Process_Setup_UI_linearBackwardLayer_Changed();

    def Process_Setup_UI_linearBackwardApply_Button_Clicked(self):
        if len(self.current_Process_Linear_Backward_List) < 2:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "At least, there are two layers in the list.");
            return;

        first_Layer = self.current_Process_Linear_Backward_List[0];
        last_Layer = self.current_Process_Linear_Backward_List[-1];

        if not self.Process_Setup_UI_Layer_Error_Calculated_Check(first_Layer):
            if self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.isChecked():
                if not self.Process_Setup_UI_Layer_Error_Stroaged_Check(first_Layer):
                    raise Exception("Linear Backward Error Process");                
                self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [first_Layer], None, None));
            elif self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.isChecked():
                self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Sigmoid, [first_Layer], None, None));
            elif self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.isChecked():
                self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Softmax, [first_Layer], None, None));
        self.current_Process_Order_List.append((Order_Code.Error_Send, [first_Layer, self.current_Process_Linear_Backward_List[1]], None, None));        
        for layer_Index in range(1, len(self.current_Process_Linear_Backward_List) - 2):
            layer = self.current_Process_Linear_Backward_List[layer_Index];
            self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Error_Send, [layer, self.current_Process_Linear_Backward_List[layer_Index + 1]], None, None));
        self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [self.current_Process_Linear_Backward_List[-2]], None, None));
        if self.process_Setup_UI.linearBackwardLastLayerTypeHidden_RadioButton.isChecked():
            self.current_Process_Order_List.append((Order_Code.Error_Send, [self.current_Process_Linear_Backward_List[-2], last_Layer], None, None));
            self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [last_Layer], None, None));

        for index in reversed(range(1, len(self.current_Process_Linear_Backward_List))):
            from_Layer_Name = self.current_Process_Linear_Backward_List[index];
            to_Layer_Name = self.current_Process_Linear_Backward_List[index - 1];
            connection = self.simulator.Extract_Connection(from_Layer_Name, to_Layer_Name);
            self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [connection], None));

        if self.process_Setup_UI.linearBackwardLastLayerTypeInput_RadioButton.isChecked():
            for layer in self.current_Process_Linear_Backward_List[:-1]:
                self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [layer], None, None));        
        elif self.process_Setup_UI.linearBackwardLastLayerTypeHidden_RadioButton.isChecked():
            for layer in self.current_Process_Linear_Backward_List:
                self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [layer], None, None));

        self.current_Process_Linear_Backward_List = [];    
        
        self.Process_Setup_UI_Order_and_Control_Changed();
        self.Process_Setup_UI_linearBackwardLayer_Changed();

    def Process_Setup_UI_linearEndInitialize_Button_Clicked(self):
        self.Process_Setup_UI_customEndInitialize_Button_Clicked();

    def Process_Setup_UI_customEndInitialize_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.End_and_Initialize, None, None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customInputLayerActivationInsert_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Input_Layer_Acitvation_Insert, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();
    
    def Process_Setup_UI_customLayerActivationSend_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Activation_Send, [self.process_Setup_UI.customLayer1_ComboBox.currentText(), self.process_Setup_UI.customLayer2_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customLayerActivationCalculationSigmoid_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Sigmoid, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customLayerActivationCalculationSoftmax_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Activation_Calculation_Softmax, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customLayerActivationCalculationReLU_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Activation_Calculation_ReLU, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customLayerInitialize_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Layer_Initialize, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customOutputLayerErrorCalculationSigmoid_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Sigmoid, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customOutputLayerErrorCalculationSoftmax_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Output_Layer_Error_Calculation_Softmax, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customHiddenLayerErrorCalculationSigmoid_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customHiddenLayerErrorCalculationReLU_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Hidden_Layer_Error_Calculation_ReLU, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customErrorSend_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Error_Send, [self.process_Setup_UI.customLayer1_ComboBox.currentText(), self.process_Setup_UI.customLayer2_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();    

    def Process_Setup_UI_customActivationExtract_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Activation_Extract, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customBiasRenewal_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Bias_Renewal, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();
        
    def Process_Setup_UI_customWeightRenewal_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Weight_Renewal, None, [self.process_Setup_UI.customConnection1_ComboBox.currentText()], None));
        self.Process_Setup_UI_Order_and_Control_Changed();
    
    def Process_Setup_UI_customLayerDuplicate_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Layer_Duplication, [self.process_Setup_UI.customLayer1_ComboBox.currentText(), self.process_Setup_UI.customLayer2_ComboBox.currentText()], None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customConnectionDuplicate_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Connection_Duplication, None, [self.process_Setup_UI.customConnection1_ComboBox.currentText(), self.process_Setup_UI.customConnection2_ComboBox.currentText()], None));
        self.Process_Setup_UI_Order_and_Control_Changed();  

    def Process_Setup_UI_customTransposedConnectionDuplicate_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Transposed_Connection_Duplication, None, [self.process_Setup_UI.customConnection1_ComboBox.currentText(), self.process_Setup_UI.customConnection2_ComboBox.currentText()], None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customBiasEqualizationAdd_Button_Clicked(self):
        selected_Layer = self.process_Setup_UI.customBiasEqualizationLayer_ComboBox.currentText();
        if selected_Layer == "":
            self.process_Setup_UI.customBiasEqualizationLayer_ComboBox.setFocus();
            return;
        elif selected_Layer in self.current_Process_Custom_Bias_Equalization_Layer_List:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "This layer is already in the list");
            return;

        self.current_Process_Custom_Bias_Equalization_Layer_List.append(selected_Layer);
    
        self.Process_Setup_UI_customBiasEqualizationLayer_Changed();

    def Process_Setup_UI_customBiasEqualizationDelete_Button_Clicked(self):
        selected_Layer_Index = self.process_Setup_UI.customBiasEqualization_ListWidget.currentRow();
        if selected_Layer_Index < 0:
            return;

        del self.current_Process_Custom_Bias_Equalization_Layer_List[selected_Layer_Index];

        self.Process_Setup_UI_customBiasEqualizationLayer_Changed();

    def Process_Setup_UI_customBiasEqualization_Button_Clicked(self):
        if len(self.current_Process_Custom_Bias_Equalization_Layer_List) < 2:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "At least, there are two layers in the list.");
            return;

        self.current_Process_Order_List.append((Order_Code.Bias_Equalization, self.current_Process_Custom_Bias_Equalization_Layer_List, None, None));        
        self.current_Process_Custom_Bias_Equalization_Layer_List = [];

        self.Process_Setup_UI_Order_and_Control_Changed();
        
    def Process_Setup_UI_customWeightEqualizationConnectionAdd_Button_Clicked(self):
        selected_Connection = self.process_Setup_UI.customWeightEqualizationConnection_ComboBox.currentText()
        if selected_Connection == "":
            self.process_Setup_UI.customWeightEqualizationConnection_ComboBox.setFocus();
            return;
        elif selected_Connection in self.current_Process_Custom_Weight_Equalization_Connection_List:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "This connection is already in the list");
            return;

        self.current_Process_Custom_Weight_Equalization_Connection_List.append(selected_Connection);
    
        self.Process_Setup_UI_customWeightEqualizationConnection_Changed();
        
    def Process_Setup_UI_customWeightEqualizationConnectionDelete_Button_Clicked(self):
        selected_Connection_Index = self.process_Setup_UI.customWeightEqualization_ListWidget.currentRow();
        if selected_Connection_Index < 0:
            return;

        del self.current_Process_Custom_Weight_Equalization_Connection_List[selected_Connection_Index];

        self.Process_Setup_UI_customWeightEqualizationConnection_Changed();
        
    def Process_Setup_UI_customWeightEqualization_Button_Clicked(self):
        if len(self.current_Process_Custom_Weight_Equalization_Connection_List) < 2:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "At least, there are two connections in the list.");
            return;

        self.current_Process_Order_List.append((Order_Code.Weight_Equalization, None, self.current_Process_Custom_Weight_Equalization_Connection_List, None));        
        self.current_Process_Custom_Weight_Equalization_Connection_List = [];

        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customCycleMaker_Button_Clicked(self):
        self.current_Process_Order_List.append((Order_Code.Cycle_Maker, None, None, None));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customUniformRandomActivationInsert_Button_Clicked(self):
        if self.process_Setup_UI.customRandomActivationCriteria_LineEdit.text() == "":
            return;

        self.current_Process_Order_List.append((Order_Code.Uniform_Random_Activation_Insert, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, [float(self.process_Setup_UI.customRandomActivationCriteria_LineEdit.text())]));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customNormalRandomActivationInsert_Button_Clicked(self):
        if self.process_Setup_UI.customRandomActivationCriteria_LineEdit.text() == "":
            return;

        self.current_Process_Order_List.append((Order_Code.Normal_Random_Activation_Insert, [self.process_Setup_UI.customLayer1_ComboBox.currentText()], None, [float(self.process_Setup_UI.customRandomActivationCriteria_LineEdit.text())]));
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_customLayer1_ComboBox_Current_Index_Changed(self):
        self.process_Setup_UI.customLayer2_ComboBox.setCurrentIndex(0);

        if not self.process_Setup_UI.customLayer1_ComboBox.currentIndex() > 0:
            return;
            
        self.process_Setup_UI.customLayer2_ComboBox.setEnabled(True);            
        self.process_Setup_UI.customConnection1_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.customConnection2_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.customConnection2_ComboBox.setEnabled(False);

        self.Process_Setup_UI_customTap_Enable();

    def Process_Setup_UI_customLayer2_ComboBox_Current_Index_Changed(self):
        if not self.process_Setup_UI.customLayer2_ComboBox.currentIndex() > 0:
            return;

        self.Process_Setup_UI_customTap_Enable();
            
    def Process_Setup_UI_customConnection1_ComboBox_Current_Index_Changed(self):
        self.process_Setup_UI.customConnection2_ComboBox.setCurrentIndex(0);

        if not self.process_Setup_UI.customConnection1_ComboBox.currentIndex() > 0:
            return;
            
        self.process_Setup_UI.customConnection2_ComboBox.setEnabled(True);            
        self.process_Setup_UI.customLayer1_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.customLayer2_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.customLayer2_ComboBox.setEnabled(False);

        self.Process_Setup_UI_customTap_Enable();

    def Process_Setup_UI_customConnection2_ComboBox_Current_Index_Changed(self):
        if not self.process_Setup_UI.customConnection1_ComboBox.currentIndex() > 0:
            return;

        self.Process_Setup_UI_customTap_Enable();

    def Process_Setup_UI_processSave_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getSaveFileName(filter="Process File for HNet (*.HNet_Process)")[0]
        if file_Path != "":
            self.simulator.Process_Save(file_Path);

    def Process_Setup_UI_processLoad_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getOpenFileName(filter="Process File for HNet (*.HNet_Process)")[0];
        if file_Path == "":
            return;

        load_Result = self.simulator.Process_Load(file_Path);

        if not load_Result:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "Current structure is not compatiable with this process file.");

        self.Process_Setup_UI_Process_Changed();

    def Process_Setup_UI_exit_Button_Clicked(self):
        self.process_Setup_Dialog.hide();
        if len(self.simulator.process_Dict) > 1:
            self.main_UI.processLock_Button.setEnabled(True);
        else:
            self.main_UI.processLock_Button.setEnabled(False);
        self.Main_UI_Simulator_Changed();
        self.main_Dialog.exec_();
    
    def Process_Setup_UI_ComboBox_Add_Item(self):
        self.process_Setup_UI.bpInputLayer_ComboBox.clear();
        self.process_Setup_UI.bpttOutputLayer_ComboBox.clear();
        self.process_Setup_UI.srnInputLayer_ComboBox.clear();
        self.process_Setup_UI.srnContextLayer_ComboBox.clear();
        self.process_Setup_UI.linearForwardLayer_ComboBox.clear();
        self.process_Setup_UI.linearBackwardLayer_ComboBox.clear();
        self.process_Setup_UI.customLayer1_ComboBox.clear();
        self.process_Setup_UI.customLayer2_ComboBox.clear();
        self.process_Setup_UI.customBiasEqualizationLayer_ComboBox.clear();    
    
        self.process_Setup_UI.customConnection1_ComboBox.clear();
        self.process_Setup_UI.customConnection2_ComboBox.clear();
        self.process_Setup_UI.customWeightEqualizationConnection_ComboBox.clear();

        self.process_Setup_UI.bpInputLayer_ComboBox.addItem("");
        self.process_Setup_UI.bpttOutputLayer_ComboBox.addItem("");
        self.process_Setup_UI.srnInputLayer_ComboBox.addItem("");
        self.process_Setup_UI.srnContextLayer_ComboBox.addItem("");
        self.process_Setup_UI.linearForwardLayer_ComboBox.addItem("");
        self.process_Setup_UI.linearBackwardLayer_ComboBox.addItem("");
        self.process_Setup_UI.customLayer1_ComboBox.addItem("");
        self.process_Setup_UI.customLayer2_ComboBox.addItem("");
        self.process_Setup_UI.customBiasEqualizationLayer_ComboBox.addItem("");    
        self.process_Setup_UI.customConnection1_ComboBox.addItem("");
        self.process_Setup_UI.customConnection2_ComboBox.addItem("");
        self.process_Setup_UI.customWeightEqualizationConnection_ComboBox.addItem("");

        for layer in self.simulator.layer_Information_Dict.keys():
            self.process_Setup_UI.bpInputLayer_ComboBox.addItem(layer);
            self.process_Setup_UI.bpttOutputLayer_ComboBox.addItem(layer);
            self.process_Setup_UI.srnInputLayer_ComboBox.addItem(layer);
            self.process_Setup_UI.srnContextLayer_ComboBox.addItem(layer);
            self.process_Setup_UI.linearForwardLayer_ComboBox.addItem(layer);
            self.process_Setup_UI.linearBackwardLayer_ComboBox.addItem(layer);
            self.process_Setup_UI.customLayer1_ComboBox.addItem(layer);
            self.process_Setup_UI.customLayer2_ComboBox.addItem(layer);
            self.process_Setup_UI.customBiasEqualizationLayer_ComboBox.addItem(layer);
        
        for connection in self.simulator.connection_Information_Dict.keys():
            self.process_Setup_UI.customConnection1_ComboBox.addItem(connection);
            self.process_Setup_UI.customConnection2_ComboBox.addItem(connection);
            self.process_Setup_UI.customWeightEqualizationConnection_ComboBox.addItem(connection);

    def Process_Setup_UI_Viewing_Mode_Widget_Enable(self):
        self.process_Setup_UI.processName_LineEdit.setEnabled(True);
        self.process_Setup_UI.process_ListWidget.setEnabled(True);
        self.process_Setup_UI.processDelete_Button.setEnabled(True);
        self.process_Setup_UI.processMaking_Button.setEnabled(True);
        self.process_Setup_UI.processModify_Button.setEnabled(True);
        self.process_Setup_UI.processEnd_Button.setEnabled(False);
        
        self.process_Setup_UI.layerOn_Button.setEnabled(False);
        self.process_Setup_UI.layerOff_Button.setEnabled(False);
        self.process_Setup_UI.layerDamage_Button.setEnabled(False);
        self.process_Setup_UI.layerDamageSD_LineEdit.setEnabled(False);

        self.process_Setup_UI.connectionOn_Button.setEnabled(False);
        self.process_Setup_UI.connectionOff_Button.setEnabled(False);
        self.process_Setup_UI.connectionDamage_Button.setEnabled(False);
        self.process_Setup_UI.connectionDamageSD_LineEdit.setEnabled(False);

        self.process_Setup_UI.orderDelete_Button.setEnabled(False);
        self.process_Setup_UI.orderUp_Button.setEnabled(False);
        self.process_Setup_UI.orderDown_Button.setEnabled(False);

        self.process_Setup_UI.order_TapWidget.setEnabled(False);

    def Process_Setup_UI_Making_Mode_Widget_Enable(self):
        self.process_Setup_UI.processName_LineEdit.setEnabled(False);
        self.process_Setup_UI.process_ListWidget.setEnabled(False);
        self.process_Setup_UI.processDelete_Button.setEnabled(False);
        self.process_Setup_UI.processMaking_Button.setEnabled(False);
        self.process_Setup_UI.processModify_Button.setEnabled(False);
        self.process_Setup_UI.processEnd_Button.setEnabled(True);
        
        self.process_Setup_UI.layerOn_Button.setEnabled(True);
        self.process_Setup_UI.layerOff_Button.setEnabled(True);
        self.process_Setup_UI.layerDamage_Button.setEnabled(True);
        self.process_Setup_UI.layerDamageSD_LineEdit.setEnabled(True);

        self.process_Setup_UI.connectionOn_Button.setEnabled(True);
        self.process_Setup_UI.connectionOff_Button.setEnabled(True);
        self.process_Setup_UI.connectionDamage_Button.setEnabled(True);
        self.process_Setup_UI.connectionDamageSD_LineEdit.setEnabled(True);

        self.process_Setup_UI.orderDelete_Button.setEnabled(True);
        self.process_Setup_UI.orderUp_Button.setEnabled(True);
        self.process_Setup_UI.orderDown_Button.setEnabled(True);

        self.process_Setup_UI.order_TapWidget.setEnabled(True);
    
    
    def Process_Setup_UI_Process_Changed(self):
        self.process_Setup_UI.processName_LineEdit.setText("");
        self.process_Setup_UI.process_ListWidget.clear();

        for key in self.simulator.process_Dict.keys():            
            self.process_Setup_UI.process_ListWidget.addItem(key);

        self.process_Setup_UI.bpInputLayer_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.bpttInputLayer_LineEdit.setText("");
        self.process_Setup_UI.bpttHiddenLayer_LineEdit.setText("");
        self.process_Setup_UI.bpttTick_LineEdit.setText("");
        self.process_Setup_UI.bpttOutputLayer_ComboBox.setCurrentIndex(0);        
        self.process_Setup_UI.srnInputLayer_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.srnContextLayer_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.linearForwardLayer_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.linearBackwardLayer_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.customLayer1_ComboBox.setCurrentIndex(0);
        self.process_Setup_UI.customConnection1_ComboBox.setCurrentIndex(0);        
        
        self.Process_Setup_UI_Order_and_Control_Changed();

    def Process_Setup_UI_Order_and_Control_Changed(self):
        self.process_Setup_UI.order_ListWidget.clear();
        self.process_Setup_UI.layerControl_ListWidget.clear();
        self.process_Setup_UI.connectionControl_ListWidget.clear();        

        #Order_List
        for order_Code, layer_Name_List, connection_Name_List, order_Variable_List in self.current_Process_Order_List:
            if not order_Variable_List is None:
                variable_Text = " Vals:(" + ", ".join([str(x) for x in order_Variable_List]) + ")";
            else:
                variable_Text = "";

            if not layer_Name_List is None:                
                self.process_Setup_UI.order_ListWidget.addItem(str(order_Code)[11:] + " (" + ", ".join([layer for layer in layer_Name_List]) + ")" + variable_Text);
            elif not connection_Name_List is None:                
                self.process_Setup_UI.order_ListWidget.addItem(str(order_Code)[11:] + " (" + ", ".join([connection for connection in connection_Name_List]) + ")" + variable_Text);
            else:
                self.process_Setup_UI.order_ListWidget.addItem(str(order_Code)[11:] + variable_Text);            
            
        #Layer_Control_List
        for layer in self.current_Process_Layer_Control_Dict.keys():
            if self.current_Process_Layer_Control_Dict[layer][0] == Damage_Type.On:
                self.process_Setup_UI.layerControl_ListWidget.addItem(str(layer) + " → On");
            elif self.current_Process_Layer_Control_Dict[layer][0] == Damage_Type.Off:
                self.process_Setup_UI.layerControl_ListWidget.addItem(str(layer) + " → Off");
            elif self.current_Process_Layer_Control_Dict[layer][0] == Damage_Type.Damaged:
                self.process_Setup_UI.layerControl_ListWidget.addItem(str(layer) + " → Damaged(" + str(self.current_Process_Layer_Control_Dict[layer][1]) + ")");
            
        #Connection_Control_List
        for connection in self.current_Process_Connection_Control_Dict.keys():
            if self.current_Process_Connection_Control_Dict[connection][0] == Damage_Type.On:
                self.process_Setup_UI.connectionControl_ListWidget.addItem(str(connection) + " → On");
            elif self.current_Process_Connection_Control_Dict[connection][0] == Damage_Type.Off:
                self.process_Setup_UI.connectionControl_ListWidget.addItem(str(connection) + " → Off");
            elif self.current_Process_Connection_Control_Dict[connection][0] == Damage_Type.Damaged:
                self.process_Setup_UI.connectionControl_ListWidget.addItem(str(connection) + " → Damaged(" + str(self.current_Process_Connection_Control_Dict[connection][1]) + ")");

        self.Process_Setup_UI_customBiasEqualizationLayer_Changed();
        self.Process_Setup_UI_customWeightEqualizationConnection_Changed();
        self.Process_Setup_UI_customRandomAcitvation_Inserted();
        self.Process_Setup_UI_bpTap_Enable();
        self.Process_Setup_UI_bpttTap_Enable();
        self.Process_Setup_UI_srnTap_Enable();
        self.Process_Setup_UI_linearTap_Enable();
        self.Process_Setup_UI_customTap_Enable();
    
    def Process_Setup_UI_linearForwardLayer_Changed(self):
        self.process_Setup_UI.linearForwardLayer_ComboBox.clear();
        self.process_Setup_UI.linearForwardLayer_ListWidget.clear();

        if len(self.current_Process_Linear_Forward_List) == 0:
            self.process_Setup_UI.linearForwardLayer_ComboBox.addItem("");
            for layer in self.simulator.layer_Information_Dict.keys():
                self.process_Setup_UI.linearForwardLayer_ComboBox.addItem(layer);
        else:            
            last_Layer = self.current_Process_Linear_Forward_List[-1];
            extract_Information = self.simulator.Extract_Connection_List(last_Layer);
            for connection, layer in extract_Information:
                self.process_Setup_UI.linearForwardLayer_ComboBox.addItem(layer);
            for layer in self.current_Process_Linear_Forward_List:
                self.process_Setup_UI.linearForwardLayer_ListWidget.addItem(layer);

        self.Process_Setup_UI_linearTap_Enable();

    def Process_Setup_UI_linearBackwardLayer_Changed(self):
        self.process_Setup_UI.linearBackwardLayer_ComboBox.clear();
        self.process_Setup_UI.linearBackwardLayer_ListWidget.clear();

        if len(self.current_Process_Linear_Backward_List) == 0:
            self.process_Setup_UI.linearBackwardLayer_ComboBox.addItem("");
            for layer in self.simulator.layer_Information_Dict.keys():
                if self.Process_Setup_UI_Layer_Activated_Check(layer):
                    self.process_Setup_UI.linearBackwardLayer_ComboBox.addItem(layer);
        else:
            last_Layer = self.current_Process_Linear_Backward_List[-1];
            for layer in self.simulator.layer_Information_Dict.keys():
                if self.simulator.Extract_Connection(layer, last_Layer) and self.Process_Setup_UI_Layer_Activated_Check(layer):
                    self.process_Setup_UI.linearBackwardLayer_ComboBox.addItem(layer);
            for layer in self.current_Process_Linear_Backward_List:
                self.process_Setup_UI.linearBackwardLayer_ListWidget.addItem(layer);

        self.Process_Setup_UI_linearTap_Enable();

    def Process_Setup_UI_customBiasEqualizationLayer_Changed(self):
        self.process_Setup_UI.customBiasEqualization_ListWidget.clear();

        for layer in self.current_Process_Custom_Bias_Equalization_Layer_List:
            self.process_Setup_UI.customBiasEqualization_ListWidget.addItem(layer);

    def Process_Setup_UI_customWeightEqualizationConnection_Changed(self):
        self.process_Setup_UI.customWeightEqualization_ListWidget.clear();

        for connection in self.current_Process_Custom_Weight_Equalization_Connection_List:
            self.process_Setup_UI.customWeightEqualization_ListWidget.addItem(connection);    
    
    def Process_Setup_UI_customRandomAcitvation_Inserted(self):
        self.process_Setup_UI.customRandomActivationCriteria_LineEdit.setText("");

    def Process_Setup_UI_bpTap_Enable(self):
        if len(self.current_Process_Order_List) > 0:
            self.process_Setup_UI.bp_Tap.setEnabled(False);
        else:
            self.process_Setup_UI.bp_Tap.setEnabled(True);

    def Process_Setup_UI_bpttTap_Enable(self):
        if len(self.current_Process_Order_List) > 0:
            self.process_Setup_UI.bptt_Tap.setEnabled(False);
        else:
            self.process_Setup_UI.bptt_Tap.setEnabled(True);

    def Process_Setup_UI_srnTap_Enable(self):
        if len(self.current_Process_Order_List) > 0:
            self.process_Setup_UI.srn_Tap.setEnabled(False);
        else:
            self.process_Setup_UI.srn_Tap.setEnabled(True);

    def Process_Setup_UI_linearTap_Enable(self):
        if len(self.current_Process_Linear_Forward_List) > 0:
            first_Layer = self.current_Process_Linear_Forward_List[0];
            if not self.Process_Setup_UI_Layer_Activation_Stroaged_Check(first_Layer) and not self.Process_Setup_UI_Layer_Activated_Check(first_Layer):
                self.process_Setup_UI.linearForwardFirstLayerTypeInput_RadioButton.setChecked(True);
                self.process_Setup_UI.linearForwardFirstLayerTypeHidden_RadioButton.setChecked(False);
                self.process_Setup_UI.linearForwardFirstLayerTypeHidden_RadioButton.setEnabled(False);
            else:
                self.process_Setup_UI.linearForwardFirstLayerTypeHidden_RadioButton.setEnabled(True);
        else:
            self.process_Setup_UI.linearForwardFirstLayerTypeHidden_RadioButton.setEnabled(True);

        if len(self.current_Process_Linear_Backward_List) > 0:
            first_Layer = self.current_Process_Linear_Backward_List[0];
            all_False_Check = True;
            if not self.Process_Setup_UI_Layer_Error_Stroaged_Check(first_Layer):
                
                self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.setEnabled(False);
            else:
                self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.setEnabled(True);
                all_False_Check = False;            
            if not self.Process_Setup_UI_Layer_Activated_Check(first_Layer) and not self.Process_Setup_UI_Layer_Error_Calculated_Check(first_Layer):
                self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.setEnabled(False);
                self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.setEnabled(False);
            else:
                self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.setEnabled(True);
                self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.setEnabled(True);
                all_False_Check = False;
            
            if all_False_Check:                
                self.process_Setup_UI.linearBackwardApply_Button.setEnabled(False);
            else:
                if self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.isChecked() and not self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.isEnabled():  
                    self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.setChecked(False);
                    self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.setChecked(True);
                if self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.isChecked() and not self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.isEnabled():
                    self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.setChecked(False);
                    self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.setChecked(True);
                if self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.isChecked() and not self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.isEnabled():
                    self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.setChecked(False);
                    self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.setChecked(True);
                
                self.process_Setup_UI.linearBackwardApply_Button.setEnabled(True);
        else:
            self.process_Setup_UI.linearBackwardFirstLayerTypeHidden_RadioButton.setEnabled(True);
            self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSigmoid_RadioButton.setEnabled(True);
            self.process_Setup_UI.linearBackwardFirstLayerTypeOutputSoftmax_RadioButton.setEnabled(True);
            self.process_Setup_UI.linearBackwardApply_Button.setEnabled(True);

        if len(self.current_Process_Order_List) > 0 and self.current_Process_Order_List[-1][0] == Order_Code.End_and_Initialize:            
            self.process_Setup_UI.linear_Tap.setEnabled(False);
        else:
            self.process_Setup_UI.linear_Tap.setEnabled(True);
        
    def Process_Setup_UI_customTap_Enable(self):
        if len(self.current_Process_Order_List) > 0 and self.current_Process_Order_List[-1][0] == Order_Code.End_and_Initialize:
            self.process_Setup_UI.custom_Tap.setEnabled(False);
            return;
        else:
            self.process_Setup_UI.custom_Tap.setEnabled(True);

        self.process_Setup_UI.customEndInitialize_Button.setEnabled(False);
        self.process_Setup_UI.customInputLayerActivationInsert_Button.setEnabled(False);
        self.process_Setup_UI.customLayerActivationSend_Button.setEnabled(False);
        self.process_Setup_UI.customLayerActivationCalculationSigmoid_Button.setEnabled(False);
        self.process_Setup_UI.customLayerActivationCalculationSoftmax_Button.setEnabled(False);
        self.process_Setup_UI.customLayerActivationCalculationReLU_Button.setEnabled(False);
        self.process_Setup_UI.customLayerInitialize_Button.setEnabled(False);
        self.process_Setup_UI.customOutputLayerErrorCalculationSigmoid_Button.setEnabled(False);
        self.process_Setup_UI.customOutputLayerErrorCalculationSoftmax_Button.setEnabled(False);
        self.process_Setup_UI.customHiddenLayerErrorCalculationSigmoid_Button.setEnabled(False);
        self.process_Setup_UI.customHiddenLayerErrorCalculationReLU_Button.setEnabled(False);
        self.process_Setup_UI.customActivationExtract_Button.setEnabled(False);
        self.process_Setup_UI.customErrorSend_Button.setEnabled(False);
        self.process_Setup_UI.customBiasRenewal_Button.setEnabled(False);
        self.process_Setup_UI.customWeightRenewal_Button.setEnabled(False);
        self.process_Setup_UI.customLayerDuplicate_Button.setEnabled(False);
        self.process_Setup_UI.customConnectionDuplicate_Button.setEnabled(False);
        self.process_Setup_UI.customTransposedConnectionDuplicate_Button.setEnabled(False);
        self.process_Setup_UI.customCycleMaker_Button.setEnabled(False);
        self.process_Setup_UI.customBiasEqualization_GroupBox.setEnabled(False);
        self.process_Setup_UI.customWeightEqualization_GroupBox.setEnabled(False);        
        self.process_Setup_UI.randomizeActivationInsert_GroupBox.setEnabled(False);

        #Always True
        self.process_Setup_UI.customEndInitialize_Button.setEnabled(True);
        self.process_Setup_UI.customCycleMaker_Button.setEnabled(True);
        self.process_Setup_UI.customBiasEqualization_GroupBox.setEnabled(True);
        self.process_Setup_UI.customWeightEqualization_GroupBox.setEnabled(True);

        #Layer
        #None, Initial, Stroaged, Activated, CleanupActivated, ConnectedLayer_ErrorCalculated, ErrorCalculated Status        
        if self.process_Setup_UI.customLayer1_ComboBox.currentIndex() > 0:
            selected_Layer1 = self.process_Setup_UI.customLayer1_ComboBox.currentText();

            self.process_Setup_UI.customInputLayerActivationInsert_Button.setEnabled(True);
            self.process_Setup_UI.customLayerInitialize_Button.setEnabled(True);
            self.process_Setup_UI.randomizeActivationInsert_GroupBox.setEnabled(True);
            if self.process_Setup_UI.customLayer2_ComboBox.currentIndex() > 0:
                selected_Layer2 = self.process_Setup_UI.customLayer2_ComboBox.currentText();
                if self.simulator.layer_Information_Dict[selected_Layer1]["Unit"] == self.simulator.layer_Information_Dict[selected_Layer2]["Unit"]:
                    self.process_Setup_UI.customLayerDuplicate_Button.setEnabled(True);

            if self.Process_Setup_UI_Layer_Activation_Stroaged_Check(selected_Layer1):
                self.process_Setup_UI.customLayerActivationCalculationSigmoid_Button.setEnabled(True);
                self.process_Setup_UI.customLayerActivationCalculationSoftmax_Button.setEnabled(True);
                self.process_Setup_UI.customLayerActivationCalculationReLU_Button.setEnabled(True);

            if self.Process_Setup_UI_Layer_Activated_Check(selected_Layer1):                
                self.process_Setup_UI.customOutputLayerErrorCalculationSigmoid_Button.setEnabled(True);
                self.process_Setup_UI.customOutputLayerErrorCalculationSoftmax_Button.setEnabled(True);
                self.process_Setup_UI.customActivationExtract_Button.setEnabled(True);
                if self.process_Setup_UI.customLayer2_ComboBox.currentIndex() > 0:
                    self.process_Setup_UI.customLayerActivationSend_Button.setEnabled(True);

            if self.Process_Setup_UI_Layer_Error_Stroaged_Check(selected_Layer1):
                self.process_Setup_UI.customHiddenLayerErrorCalculationSigmoid_Button.setEnabled(True);
                self.process_Setup_UI.customHiddenLayerErrorCalculationReLU_Button.setEnabled(True);
                        
            if self.Process_Setup_UI_Layer_Error_Calculated_Check(selected_Layer1):
                self.process_Setup_UI.customBiasRenewal_Button.setEnabled(True);
                if self.process_Setup_UI.customLayer2_ComboBox.currentIndex() > 0:
                    self.process_Setup_UI.customErrorSend_Button.setEnabled(True);

        if self.process_Setup_UI.customConnection1_ComboBox.currentIndex() > 0:
            selected_Connection1 = self.process_Setup_UI.customConnection1_ComboBox.currentText();

            if self.process_Setup_UI.customConnection2_ComboBox.currentIndex() > 0:
                selected_Connection2 = self.process_Setup_UI.customConnection2_ComboBox.currentText();
                if self.simulator.connection_Information_Dict[selected_Connection1]["From_Layer_Unit"] == self.simulator.connection_Information_Dict[selected_Connection2]["From_Layer_Unit"] and self.simulator.connection_Information_Dict[selected_Connection1]["To_Layer_Unit"] == self.simulator.connection_Information_Dict[selected_Connection2]["To_Layer_Unit"]:
                    self.process_Setup_UI.customConnectionDuplicate_Button.setEnabled(True);
                elif self.simulator.connection_Information_Dict[selected_Connection1]["From_Layer_Unit"] == self.simulator.connection_Information_Dict[selected_Connection2]["To_Layer_Unit"] and self.simulator.connection_Information_Dict[selected_Connection1]["To_Layer_Unit"] == self.simulator.connection_Information_Dict[selected_Connection2]["From_Layer_Unit"]:
                    self.process_Setup_UI.customTransposedConnectionDuplicate_Button.setEnabled(True);

            if self.Process_Setup_UI_Connection_From_Activated_To_Error_Calculated_Check(selected_Connection1):
                self.process_Setup_UI.customWeightRenewal_Button.setEnabled(True);            

    def Process_Setup_UI_Layer_Activation_Stroaged_Check(self, layer, check_Index = None):
        if check_Index == None:
            check_Index = len(self.current_Process_Order_List); 

        activation_Storaged = False;
        current_Index = 0;
        while current_Index < check_Index:
            order_Code, layer_List, connection_List, order_Variable_List = self.current_Process_Order_List[current_Index];
            if order_Code in [Order_Code.Activation_Send] and layer == layer_List[1]:
                activation_Storaged = True;
            elif order_Code in [Order_Code.Layer_Duplication] and layer == layer_List[1]:
                activation_Storaged = self.Process_Setup_UI_Layer_Activation_Stroaged_Check(layer_List[0], current_Index);
            elif order_Code == Order_Code.Layer_Initialize and layer == layer_List[0]:
                activation_Storaged = False;
            current_Index += 1;
        return activation_Storaged;

    def Process_Setup_UI_Layer_Activated_Check(self, layer, check_Index = None):
        if check_Index == None:
            check_Index = len(self.current_Process_Order_List);

        activated = False;
        current_Index = 0;
        while current_Index < check_Index:
            order_Code, layer_List, connection_List, order_Variable_List = self.current_Process_Order_List[current_Index];
            if order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Activation_Calculation_Sigmoid, Order_Code.Activation_Calculation_Softmax, Order_Code.Activation_Calculation_ReLU, Order_Code.Uniform_Random_Activation_Insert, Order_Code.Normal_Random_Activation_Insert] and layer == layer_List[0]:
                activated = True;
            elif order_Code in [Order_Code.Layer_Duplication] and layer == layer_List[1]:
                activated = self.Process_Setup_UI_Layer_Activated_Check(layer_List[0], current_Index);
            elif order_Code == Order_Code.Layer_Initialize and layer == layer_List[0]:
                activated = False;
            current_Index += 1;
        return activated;
    
    def Process_Setup_UI_Layer_Error_Stroaged_Check(self, layer, check_Index = None):
        if check_Index == None:
            check_Index = len(self.current_Process_Order_List); 

        error_Storaged = False;
        current_Index = 0;
        while current_Index < check_Index:
            order_Code, layer_List, connection_List, order_Variable_List = self.current_Process_Order_List[current_Index];
            if order_Code in [Order_Code.Error_Send] and layer == layer_List[1]:
                error_Storaged = True;
            elif order_Code in [Order_Code.Layer_Duplication] and layer == layer_List[1]:
                error_Storaged = self.Process_Setup_UI_Layer_Error_Stroaged_Check(layer_List[0], current_Index);
            elif order_Code == Order_Code.Layer_Initialize and layer == layer_List[0]:
                error_Storaged = False;
            current_Index += 1;
        return error_Storaged;

    def Process_Setup_UI_Layer_Error_Calculated_Check(self, layer, check_Index = None):
        if check_Index == None:
            check_Index = len(self.current_Process_Order_List); 

        errorCalculated = False;
        current_Index = 0;
        while current_Index < check_Index:
            order_Code, layer_List, connection_List, order_Variable_List = self.current_Process_Order_List[current_Index];
            if order_Code in [Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax, Order_Code.Hidden_Layer_Error_Calculation_Sigmoid, Order_Code.Hidden_Layer_Error_Calculation_ReLU] and layer == layer_List[0]:
                errorCalculated = True;
            elif order_Code in [Order_Code.Layer_Duplication] and layer == layer_List[1]:
                errorCalculated = self.Process_Setup_UI_Layer_Error_Calculated_Check(layer_List[0], current_Index);
            elif order_Code == Order_Code.Layer_Initialize and layer == layer_List[0]:
                errorCalculated = False;
            current_Index += 1;
        return errorCalculated;

    def Process_Setup_UI_Connection_From_Activated_To_Error_Calculated_Check(self, connection, check_Index = None):
        if check_Index == None:
            check_Index = len(self.current_Process_Order_List);

        from_Layer = self.simulator.connection_Information_Dict[connection]["From_Layer_Name"];
        to_Layer = self.simulator.connection_Information_Dict[connection]["To_Layer_Name"];
        return self.Process_Setup_UI_Layer_Activated_Check(from_Layer, check_Index) and self.Process_Setup_UI_Layer_Error_Calculated_Check(to_Layer, check_Index);
    
    def Process_Setup_UI_Order_Consistency_Check(self, order_Index):        
        check_Order = self.current_Process_Order_List[order_Index];
        
        always_Fine_Order_Code_List = [
            Order_Code.Input_Layer_Acitvation_Insert,
            Order_Code.Layer_Duplication,
            Order_Code.Connection_Duplication,
            Order_Code.Transposed_Connection_Duplication,
            Order_Code.Bias_Equalization,
            Order_Code.Weight_Equalization,    
            Order_Code.Layer_Initialize,
            Order_Code.Cycle_Maker,
            Order_Code.Uniform_Random_Activation_Insert,
            Order_Code.Normal_Random_Activation_Insert
        ]
        need_Activation_Stroaged_Order_Code_List = [
            Order_Code.Activation_Calculation_Sigmoid,
            Order_Code.Activation_Calculation_Softmax,
            Order_Code.Activation_Calculation_ReLU
        ]
        need_Activated_Order_Code_List = [
            Order_Code.Activation_Send,
            Order_Code.Output_Layer_Error_Calculation_Sigmoid,
            Order_Code.Output_Layer_Error_Calculation_Softmax,
            Order_Code.Activation_Extract
        ]
        need_Error_Stroaged_Order_Code_List = [
            Order_Code.Hidden_Layer_Error_Calculation_Sigmoid,
            Order_Code.Hidden_Layer_Error_Calculation_ReLU
        ]
        need_ErrorCalculated_Order_Code_List = [
            Order_Code.Error_Send,
            Order_Code.Bias_Renewal
        ]
        need_From_Activated_To_Error_Calculated_Order_Code_List = [
            Order_Code.Weight_Renewal
        ]        
        always_Impossible_Order_Code_List = [
            Order_Code.End_and_Initialize
        ]
        
        if check_Order[0] in always_Fine_Order_Code_List:            
            return True;
        elif check_Order[0] in need_Activation_Stroaged_Order_Code_List and self.Process_Setup_UI_Layer_Activation_Stroaged_Check(check_Order[1][0], order_Index):
            return True;
        elif check_Order[0] in need_Activated_Order_Code_List and self.Process_Setup_UI_Layer_Activated_Check(check_Order[1][0], order_Index):
            return True;
        elif check_Order[0] in need_ErrorCalculated_Order_Code_List and self.Process_Setup_UI_Layer_Error_Calculated_Check(check_Order[1][0], order_Index):
            return True;
        elif check_Order[0] in need_Error_Stroaged_Order_Code_List and self.Process_Setup_UI_Layer_Error_Stroaged_Check(check_Order[1][0], order_Index):
            return True;
        elif check_Order[0] in need_From_Activated_To_Error_Calculated_Order_Code_List and self.Process_Setup_UI_Connection_From_Activated_To_Error_Calculated_Check(check_Order[2][0], order_Index):
            return True;        
        else:
            return False;
            
    #End Process Setup Functions
    
    # Start Learning Setup Functions    
    def Learning_Setup_UI_learningSetupMaking_Button_Clicked(self):
        learning_Setup_Name = self.learning_Setup_UI.learningSetupName_LineEdit.text()
        if learning_Setup_Name == "":
            self.learning_Setup_UI.learningSetupName_LineEdit.setFocus();
            return;
        for learning_Setup in self.simulator.learning_Setup_List:
            if learning_Setup["Name"] == learning_Setup_Name:
                QtWidgets.QMessageBox.warning(None, 'Warning!', "There is this name in the list.");
                return;

        self.learning_Setup_UI.learningSetup_ListWidget.setCurrentRow(-1);
        
        self.Learning_Setup_UI_Current_Learning_Setup_Changed();
        self.Learning_Setup_UI_Making_Mode_Widget_Enable();
    
    def Learning_Setup_UI_learningSetupModify_Button_Clicked(self):
        selected_Index = self.learning_Setup_UI.learningSetup_ListWidget.currentRow();        
        if selected_Index < 0:
            return;

        self.learning_Setup_UI.learningSetupName_LineEdit.setText(str(self.simulator.learning_Setup_List[selected_Index]["Name"]));

        self.Learning_Setup_UI_Current_Learning_Setup_Changed();
        self.Learning_Setup_UI_Making_Mode_Widget_Enable();

    def Learning_Setup_UI_learningSetupEnd_Button_Clicked(self):
        if self.learning_Setup_UI.trainingEpoch_LineEdit.text() == "":
            self.learning_Setup_UI.trainingEpoch_LineEdit.setFocus();
            return;
        elif self.learning_Setup_UI.testTiming_LineEdit.text() == "":
            self.learning_Setup_UI.testTiming_LineEdit.setFocus();
            return;
        elif self.learning_Setup_UI.miniBatchSize_LineEdit.text() == "":
            self.learning_Setup_UI.miniBatchSize_LineEdit.setFocus();
            return;

        name = self.learning_Setup_UI.learningSetupName_LineEdit.text();
        training_Epoch = int(self.learning_Setup_UI.trainingEpoch_LineEdit.text());
        test_Timing = int(self.learning_Setup_UI.testTiming_LineEdit.text());
        mini_Batch_Size = int(self.learning_Setup_UI.miniBatchSize_LineEdit.text());
        shuffle_Mode = Shuffle_Mode(self.learning_Setup_UI.shufflingMethod_ComboBox.currentIndex());
        
        self.simulator.Learning_Setup_Assign(name, self.current_Training_Matching_List, self.current_Test_Matching_List, training_Epoch, test_Timing, mini_Batch_Size, shuffle_Mode);
        
        self.Learning_Setup_UI_Learning_Setup_Changed();
        self.Learning_Setup_UI_Viewing_Mode_Widget_Enable();        

    def Learning_Setup_UI_learningSetup_ListWidget_Current_Item_Changed(self):
        self.Learning_Setup_UI_Current_Learning_Setup_Changed();

    def Learning_Setup_UI_learningSetupDelete_Button_Clicked(self):        
        selected_Index = self.learning_Setup_UI.learningSetup_ListWidget.currentRow();
        if selected_Index < 0:
            return;

        self.simulator.Learning_Setup_Delete(selected_Index);

        self.Learning_Setup_UI_Learning_Setup_Changed();

    def Learning_Setup_UI_learningSetupUp_Button_Clicked(self):
        selected_Index = self.learning_Setup_UI.learningSetup_ListWidget.currentRow();        
        if selected_Index < 1:
            return;

        self.simulator.learning_Setup_List[selected_Index], self.simulator.learning_Setup_List[selected_Index - 1] = self.simulator.learning_Setup_List[selected_Index - 1], self.simulator.learning_Setup_List[selected_Index];

        self.Learning_Setup_UI_Learning_Setup_Changed();
        
    def Learning_Setup_UI_learningSetupDown_Button_Clicked(self, index):
        selected_Index = self.learning_Setup_UI.learningSetup_ListWidget.currentRow();
        if selected_Index < 0 and selected_Index > len(self.learning_Setup_List) - 2:
            return;
            
        self.simulator.learning_Setup_List[selected_Index], self.simulator.learning_Setup_List[selected_Index + 1] = self.simulator.learning_Setup_List[selected_Index + 1], self.simulator.learning_Setup_List[selected_Index];

        self.Learning_Setup_UI_Learning_Setup_Changed();

    def Learning_Setup_UI_trainingPatternMatchingMaking_Button_Clicked(self):
        self.Learning_Setup_UI_Training_Matching_Mode_Widget_Enable();
        self.learning_Setup_UI.trainingPatternMatching_ListWidget.setCurrentRow(-1);

    def Learning_Setup_UI_trainingPatternMatchingEnd_Button_Clicked(self):
        if self.learning_Setup_UI.trainingPatternPack_ComboBox.currentText() == "":
            self.learning_Setup_UI.trainingPatternPack_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.trainingProcess_ComboBox.currentText() == "":
            self.learning_Setup_UI.trainingProcess_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.trainingOrder_ComboBox.count() > 0:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "A pattern should be assigned to all orders which require pattern.");
            return;

        self.current_Training_Matching_List.append(self.current_Training_Matching_Information);

        self.Learning_Setup_UI_Current_Training_Matching_List_Changed();
        self.Learning_Setup_UI_Training_Viewing_Mode_Widget_Enable();

    def Learning_Setup_UI_trainingPatternMatching_ListWidget_Current_Item_Changed(self):
        self.Learning_Setup_UI_Current_Training_Matching_Changed();

    def Learning_Setup_UI_trainingPatternMatchingDelete_Button_Clicked(self):
        selected_Index = self.learning_Setup_UI.trainingPatternMatching_ListWidget.currentRow();
        if selected_Index < 0:
            return;
            
        del self.current_Training_Matching_List[selected_Index];

        self.Learning_Setup_UI_Current_Training_Matching_List_Changed();

    def Learning_Setup_UI_trainingPatternMatchingUp_Button_Clicked(self):
        selected_Index = self.learning_Setup_UI.trainingPatternMatching_ListWidget.currentRow();        
        if selected_Index < 1:
            return;

        self.current_Training_Matching_List[selected_Index], self.current_Training_Matching_List[selected_Index - 1] = self.current_Training_Matching_List[selected_Index - 1], self.current_Training_Matching_List[selected_Index];

        self.Learning_Setup_UI_Current_Training_Matching_List_Changed();

    def Learning_Setup_UI_trainingPatternMatchingDown_Button_Clicked(self):
        selected_Index = self.learning_Setup_UI.trainingPatternMatching_ListWidget.currentRow();        
        if selected_Index > len(self.current_Training_Matching_List) - 2:
            return;
            
        self.current_Training_Matching_List[selected_Index], self.current_Training_Matching_List[selected_Index + 1] = self.current_Training_Matching_List[selected_Index + 1], self.current_Training_Matching_List[selected_Index];

        self.Learning_Setup_UI_Current_Training_Matching_List_Changed();

    def Learning_Setup_UI_trainingPatternPack_ComboBox_Current_Index_Changed(self):
        self.Learning_Setup_UI_Current_Training_Matching_Pattern_Pack_and_Process_Changed();

    def Learning_Setup_UI_trainingProcess_ComboBox_Current_Index_Changed(self):        
        self.Learning_Setup_UI_Current_Training_Matching_Pattern_Pack_and_Process_Changed();

    def Learning_Setup_UI_trainingAutoAssign_Button_Clicked(self):
        if self.learning_Setup_UI.trainingPatternPack_ComboBox.currentText() == "":
            self.learning_Setup_UI.trainingPatternPack_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.trainingProcess_ComboBox.currentText() == "":
            self.learning_Setup_UI.trainingProcess_ComboBox.setFocus();
            return;

        selected_Pattern_Pack = self.simulator.pattern_Pack_Dict[self.learning_Setup_UI.trainingPatternPack_ComboBox.currentText()];
        selected_Process = self.simulator.process_Dict[self.learning_Setup_UI.trainingProcess_ComboBox.currentText()];
        pattern_Required_Order = [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax];

        assign_Fail_Index_List = [];
        curren_Cycle = 1;
        for index in range(len(selected_Process["Order_List"])):            
            order_Code, layer_List, connection_List, order_Variable_List = selected_Process["Order_List"][index];
            if order_Code == Order_Code.Cycle_Maker:
                curren_Cycle += 1;
                continue;
            elif index in self.current_Training_Matching_Information["Assign"].keys(): #Already Assigned
                continue;
            elif not order_Code in pattern_Required_Order:
                continue;
            candidate_Pattern_List1 = [];
            for pattern_Name in selected_Pattern_Pack.keys():
                if pattern_Name in ["Name", "Probability", "Cycle", "Count"]:
                    continue;
                layer_Size = self.simulator.layer_Information_Dict[layer_List[0]]["Unit"];
                pattern_Size = selected_Pattern_Pack[pattern_Name].shape[1];
                if layer_Size == pattern_Size:
                    candidate_Pattern_List1.append(pattern_Name);
            if len(candidate_Pattern_List1) == 0:
                QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no matching pattern. Auto assign was suspended. This pair cannot use together.");
                return;
            if len(candidate_Pattern_List1) == 1:
                self.current_Training_Matching_Information["Assign"][index] = candidate_Pattern_List1[0];
                continue;
            candidate_Pattern_List2 = [];
            for pattern_Name in candidate_Pattern_List1:
                if pattern_Name in layer_List[0] or layer_List[0] in pattern_Name:
                    candidate_Pattern_List2.append(pattern_Name);
            if len(candidate_Pattern_List2) == 1:
                self.current_Training_Matching_Information["Assign"][index] = candidate_Pattern_List2[0];
                continue;
            candidate_Pattern_List3 = [];
            for pattern_Name in candidate_Pattern_List1:
                if pattern_Name.strip().split("_")[-1] == str(curren_Cycle):
                    candidate_Pattern_List3.append(pattern_Name);
            if len(candidate_Pattern_List3) == 1:
                self.current_Training_Matching_Information["Assign"][index] = candidate_Pattern_List3[0];
                continue;
            candidate_Pattern_List4 = [];
            for pattern_Name in candidate_Pattern_List1:
                if pattern_Name.strip().split("_")[-1] == layer_List[0].strip().split("_")[-1]:
                    candidate_Pattern_List4.append(pattern_Name);
            if len(candidate_Pattern_List4) == 1:
                self.current_Training_Matching_Information["Assign"][index] = candidate_Pattern_List4[0];
                continue;
            else:
                assign_Fail_Index_List.append(index);

        if len(assign_Fail_Index_List) > 0:  
            QtWidgets.QMessageBox.warning(None, 'Warning!', "Some patterns could not be assined. It is recommanded that both of layer and pattern names are related with each other.");

        self.Learning_Setup_UI_Current_Training_Matching_Assign_Changed();

    def Learning_Setup_UI_trainingPatternToOrderAssign_Button_Clicked(self):
        if self.learning_Setup_UI.trainingPatternPack_ComboBox.currentText() == "":
            self.learning_Setup_UI.trainingPatternPack_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.trainingProcess_ComboBox.currentText() == "":
            self.learning_Setup_UI.trainingProcess_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.trainingPattern_ComboBox.currentText() == "":
            self.learning_Setup_UI.trainingPattern_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.trainingOrder_ComboBox.currentText()=="":
            self.learning_Setup_UI.trainingOrder_ComboBox.setFocus();
            return;
        
        pattern_Pack_Name = self.learning_Setup_UI.trainingPatternPack_ComboBox.currentText();
        process_Name = self.learning_Setup_UI.trainingProcess_ComboBox.currentText();
        order_Index = int(self.learning_Setup_UI.trainingOrder_ComboBox.currentText().split(":")[0]);
        pattern_Name = self.learning_Setup_UI.trainingPattern_ComboBox.currentText();        
        layer_Name = self.simulator.process_Dict[process_Name]["Order_List"][order_Index][1][0];

        if self.simulator.layer_Information_Dict[layer_Name]["Unit"] != self.simulator.pattern_Pack_Dict[pattern_Pack_Name][pattern_Name].shape[1]:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer unit size and pattern size are not matched. This pair cannot use together.");
            return;

        self.current_Training_Matching_Information["Assign"][order_Index] = pattern_Name;

        self.Learning_Setup_UI_Current_Training_Matching_Assign_Changed();

    def Learning_Setup_UI_trainingPatternToOrderDelete_Button_Clicked(self):
        if self.learning_Setup_UI.trainingPatternToOrderInformation_ListWidget.currentRow() < 0:
            return;
            
        del self.current_Training_Matching_Information["Assign"][int(self.learning_Setup_UI.trainingPatternToOrderInformation_ListWidget.currentItem().text().split(":")[0])];

        self.Learning_Setup_UI_Current_Training_Matching_Assign_Changed();

    def Learning_Setup_UI_testPatternMatchingMaking_Button_Clicked(self):
        self.Learning_Setup_UI_Test_Matching_Mode_Widget_Enable();
        self.learning_Setup_UI.testPatternMatching_ListWidget.setCurrentRow(-1);

    def Learning_Setup_UI_testPatternMatchingEnd_Button_Clicked(self):
        if self.learning_Setup_UI.testPatternPack_ComboBox.currentText() == "":
            self.learning_Setup_UI.testPatternPack_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.testProcess_ComboBox.currentText() == "":
            self.learning_Setup_UI.testProcess_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.testOrder_ComboBox.count() > 0:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "A pattern should be assigned to all orders which require pattern.");
            return;

        self.current_Test_Matching_List.append(self.current_Test_Matching_Information);

        self.Learning_Setup_UI_Current_Test_Matching_List_Changed();
        self.Learning_Setup_UI_Test_Viewing_Mode_Widget_Enable();

    def Learning_Setup_UI_testPatternMatching_ListWidget_Current_Item_Changed(self):
        self.Learning_Setup_UI_Current_Test_Matching_Changed();

    def Learning_Setup_UI_testPatternMatchingDelete_Button_Clicked(self):
        selected_Index = self.learning_Setup_UI.testPatternMatching_ListWidget.currentRow();
        if selected_Index < 0:
            return;

        del self.current_Test_Matching_List[selected_Index];

        self.Learning_Setup_UI_Current_Test_Matching_List_Changed();

    def Learning_Setup_UI_testPatternPack_ComboBox_Current_Index_Changed(self):
        self.Learning_Setup_UI_Current_Test_Matching_Pattern_Pack_and_Process_Changed();

    def Learning_Setup_UI_testProcess_ComboBox_Current_Index_Changed(self):        
        self.Learning_Setup_UI_Current_Test_Matching_Pattern_Pack_and_Process_Changed();

    def Learning_Setup_UI_testAutoAssign_Button_Clicked(self):        
        if self.learning_Setup_UI.testPatternPack_ComboBox.currentText() == "":
            self.learning_Setup_UI.testPatternPack_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.testProcess_ComboBox.currentText() == "":
            self.learning_Setup_UI.testProcess_ComboBox.setFocus();
            return;

        selected_Pattern_Pack = self.simulator.pattern_Pack_Dict[self.learning_Setup_UI.testPatternPack_ComboBox.currentText()];
        selected_Process = self.simulator.process_Dict[self.learning_Setup_UI.testProcess_ComboBox.currentText()];
        pattern_Required_Order = [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax];

        assign_Fail_Index_List = [];
        curren_Cycle = 1;
        for index in range(len(selected_Process["Order_List"])):            
            order_Code, layer_List, connection_List, order_Variable_List = selected_Process["Order_List"][index];
            if order_Code == Order_Code.Cycle_Maker:
                curren_Cycle += 1;
                continue;
            elif index in self.current_Test_Matching_Information["Assign"].keys(): #Already Assigned
                continue;
            elif not order_Code in pattern_Required_Order:
                continue;
            candidate_Pattern_List1 = [];
            for pattern_Name in selected_Pattern_Pack.keys():
                if pattern_Name in ["Name", "Probability", "Cycle", "Count"]:
                    continue;
                layer_Size = self.simulator.layer_Information_Dict[layer_List[0]]["Unit"];
                pattern_Size = selected_Pattern_Pack[pattern_Name].shape[1];
                if layer_Size == pattern_Size:
                    candidate_Pattern_List1.append(pattern_Name);
            if len(candidate_Pattern_List1) == 0:
                QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no matching pattern. Auto assign was suspended. This pair cannot use together.");
                return;
            if len(candidate_Pattern_List1) == 1:
                self.current_Test_Matching_Information["Assign"][index] = candidate_Pattern_List1[0];
                continue;
            candidate_Pattern_List2 = [];
            for pattern_Name in candidate_Pattern_List1:
                if pattern_Name in layer_List[0] or layer_List[0] in pattern_Name:
                    candidate_Pattern_List2.append(pattern_Name);
            if len(candidate_Pattern_List2) == 1:
                self.current_Test_Matching_Information["Assign"][index] = candidate_Pattern_List2[0];
                continue;
            candidate_Pattern_List3 = [];
            for pattern_Name in candidate_Pattern_List1:
                if pattern_Name.strip().split("_")[-1] == str(curren_Cycle):
                    candidate_Pattern_List3.append(pattern_Name);
            if len(candidate_Pattern_List3) == 1:
                self.current_Test_Matching_Information["Assign"][index] = candidate_Pattern_List3[0];
                continue;
            candidate_Pattern_List4 = [];
            for pattern_Name in candidate_Pattern_List1:
                if pattern_Name.strip().split("_")[-1] == layer_List[0].strip().split("_")[-1]:
                    candidate_Pattern_List4.append(pattern_Name);
            if len(candidate_Pattern_List4) == 1:
                self.current_Test_Matching_Information["Assign"][index] = candidate_Pattern_List4[0];
                continue;
            else:
                assign_Fail_Index_List.append(index);

        if len(assign_Fail_Index_List) > 0:  
            QtWidgets.QMessageBox.warning(None, 'Warning!', "Some patterns could not be assined. It is recommanded that both of layer and pattern names are related with each other.");

        self.Learning_Setup_UI_Current_Test_Matching_Assign_Changed();

    def Learning_Setup_UI_testPatternToOrderAssign_Button_Clicked(self):
        if self.learning_Setup_UI.testPatternPack_ComboBox.currentText() == "":
            self.learning_Setup_UI.testPatternPack_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.testProcess_ComboBox.currentText() == "":
            self.learning_Setup_UI.testProcess_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.testPattern_ComboBox.currentText() == "":
            self.learning_Setup_UI.testPattern_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.testOrder_ComboBox.currentText()=="":
            self.learning_Setup_UI.testOrder_ComboBox.setFocus();
            return;
        
        pattern_Pack_Name = self.learning_Setup_UI.testPatternPack_ComboBox.currentText();
        process_Name = self.learning_Setup_UI.testProcess_ComboBox.currentText();
        order_Index = int(self.learning_Setup_UI.testOrder_ComboBox.currentText().split(":")[0]);
        pattern_Name = self.learning_Setup_UI.testPattern_ComboBox.currentText();        
        layer_Name = self.simulator.process_Dict[process_Name]["Order_List"][order_Index][1][0];

        if self.simulator.layer_Information_Dict[layer_Name]["Unit"] != self.simulator.pattern_Pack_Dict[pattern_Pack_Name][pattern_Name].shape[1]:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer unit size and pattern size are not matched. This pair cannot use together.");
            return;

        self.current_Test_Matching_Information["Assign"][order_Index] = pattern_Name;

        self.Learning_Setup_UI_Current_Test_Matching_Assign_Changed();

    def Learning_Setup_UI_testPatternToOrderDelete_Button_Clicked(self):
        if self.learning_Setup_UI.testPatternToOrderInformation_ListWidget.currentRow() < 0:
            return;
            
        del self.current_Test_Matching_Information["Assign"][int(self.learning_Setup_UI.testPatternToOrderInformation_ListWidget.currentItem().text().split(":")[0])];

        self.Learning_Setup_UI_Current_Test_Matching_Assign_Changed();
    
    def Learning_Setup_UI_extractDataType_ComboBox_Current_Index_Changed(self):
        if Extract_Data_Type(self.learning_Setup_UI.extractDataType_ComboBox.currentIndex()) in [Extract_Data_Type.Raw_Activation, Extract_Data_Type.Semantic_Stress]:
            self.learning_Setup_UI.extractDataPattern_ComboBox.setEnabled(False);
        else:
            self.learning_Setup_UI.extractDataPattern_ComboBox.setEnabled(True);

    def Learning_Setup_UI_extractDataAssign_Button_Clicked(self):        
        if self.learning_Setup_UI.testPatternPack_ComboBox.currentText() == "":
            self.learning_Setup_UI.testPatternPack_ComboBox.setFocus();
            return;
        elif self.learning_Setup_UI.testProcess_ComboBox.currentText() == "":
            self.learning_Setup_UI.testProcess_ComboBox.setFocus();
            return;
            
        selected_Index = self.learning_Setup_UI.extractDataType_ComboBox.currentIndex();
        if Extract_Data_Type(selected_Index) in [Extract_Data_Type.Raw_Activation, Extract_Data_Type.Semantic_Stress]:
            pattern = None;
        else:
            pattern = self.learning_Setup_UI.extractDataPattern_ComboBox.currentText();

        self.current_Test_Matching_Information["Extract_Data"].append((pattern, int(self.learning_Setup_UI.extractDataOrder_ComboBox.currentText().split(":")[0]), Extract_Data_Type(selected_Index)));

        self.Learning_Setup_UI_Current_Test_Extract_Data_Changed();

    def Learning_Setup_UI_extractDataDelete_Button_Clicked(self):
        if self.learning_Setup_UI.extractData_ListWidget.currentRow() < 0:
            return;
            
        del self.current_Test_Matching_Information["Extract_Data"][self.learning_Setup_UI.extractData_ListWidget.currentRow()];

        self.Learning_Setup_UI_Current_Test_Extract_Data_Changed();
        
    def Learning_Setup_UI_save_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getSaveFileName(filter="Learning Setup File for HNet (*.HNet_Learning_Setup)")[0]

        if file_Path != "":
            self.simulator.Learning_Setup_Save(file_Path);

    def Learning_Setup_UI_load_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getOpenFileName(filter="Learning Setup File for HNet (*.HNet_Learning_Setup)")[0]

        if file_Path == "":
            return;
        
        load_Result = self.simulator.Learning_Setup_Load(file_Path);

        if not load_Result:
            QtWidgets.QMessageBox.warning(None, 'Warning!', "Current structure or process is not compatiable with this learning setup file.");

        self.Learning_Setup_UI_Learning_Setup_Changed();
        self.Learning_Setup_UI_Viewing_Mode_Widget_Enable();        

    def Learning_Setup_UI_exit_Button_Clicked(self):
        self.learning_Setup_Dialog.hide();
        if len(self.simulator.learning_Setup_List) > 0:
            self.main_UI.learning_Button.setEnabled(True);
            self.main_UI.modelSaveforMacro_Button.setEnabled(True);
            self.main_UI.patternSetup_Button.setEnabled(False);
        else:
            self.main_UI.learning_Button.setEnabled(False);
            self.main_UI.modelSaveforMacro_Button.setEnabled(False);
            self.main_UI.patternSetup_Button.setEnabled(True);
        self.Main_UI_Simulator_Changed();
        self.main_Dialog.exec_();

    def Learning_Setup_UI_ComboBox_Add_Item(self):
        self.learning_Setup_UI.trainingPatternPack_ComboBox.clear();
        self.learning_Setup_UI.testPatternPack_ComboBox.clear();
        self.learning_Setup_UI.trainingProcess_ComboBox.clear();
        self.learning_Setup_UI.testProcess_ComboBox.clear();
        
        self.learning_Setup_UI.trainingPatternPack_ComboBox.addItem("");
        self.learning_Setup_UI.testPatternPack_ComboBox.addItem("");
        for pattern_Pack in self.simulator.pattern_Pack_Dict.keys():
            self.learning_Setup_UI.trainingPatternPack_ComboBox.addItem(pattern_Pack);
            self.learning_Setup_UI.testPatternPack_ComboBox.addItem(pattern_Pack);

        self.learning_Setup_UI.trainingProcess_ComboBox.addItem("");
        self.learning_Setup_UI.testProcess_ComboBox.addItem("");
        for process in self.simulator.process_Dict.keys():
            self.learning_Setup_UI.trainingProcess_ComboBox.addItem(process);
            self.learning_Setup_UI.testProcess_ComboBox.addItem(process);

    def Learning_Setup_UI_Viewing_Mode_Widget_Enable(self):
        self.learning_Setup_UI.learningSetupName_LineEdit.setEnabled(True);
        self.learning_Setup_UI.learningSetup_ListWidget.setEnabled(True);
        self.learning_Setup_UI.learningSetupDelete_Button.setEnabled(True);
        self.learning_Setup_UI.learningSetupUp_Button.setEnabled(True);
        self.learning_Setup_UI.learningSetupDown_Button.setEnabled(True);
        self.learning_Setup_UI.learningSetupMaking_Button.setEnabled(True);
        self.learning_Setup_UI.learningSetupModify_Button.setEnabled(True);
        self.learning_Setup_UI.learningSetupEnd_Button.setEnabled(False);

        self.learning_Setup_UI.trainingEpoch_LineEdit.setEnabled(False);
        self.learning_Setup_UI.testTiming_LineEdit.setEnabled(False);
        self.learning_Setup_UI.miniBatchSize_LineEdit.setEnabled(False);
        self.learning_Setup_UI.shufflingMethod_ComboBox.setEnabled(False);

        self.learning_Setup_UI.trainingPatternMatchingMaking_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatchingDelete_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatchingUp_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatchingDown_Button.setEnabled(False);

        self.learning_Setup_UI.testPatternMatchingMaking_Button.setEnabled(False);
        self.learning_Setup_UI.testPatternMatchingDelete_Button.setEnabled(False);

    def Learning_Setup_UI_Making_Mode_Widget_Enable(self):
        self.learning_Setup_UI.learningSetupName_LineEdit.setEnabled(False);
        self.learning_Setup_UI.learningSetup_ListWidget.setEnabled(False);
        self.learning_Setup_UI.learningSetupDelete_Button.setEnabled(False);
        self.learning_Setup_UI.learningSetupUp_Button.setEnabled(False);
        self.learning_Setup_UI.learningSetupDown_Button.setEnabled(False);
        self.learning_Setup_UI.learningSetupMaking_Button.setEnabled(False);
        self.learning_Setup_UI.learningSetupModify_Button.setEnabled(False);
        self.learning_Setup_UI.learningSetupEnd_Button.setEnabled(True);

        self.learning_Setup_UI.trainingEpoch_LineEdit.setEnabled(True);
        self.learning_Setup_UI.testTiming_LineEdit.setEnabled(True);
        self.learning_Setup_UI.miniBatchSize_LineEdit.setEnabled(True);
        self.learning_Setup_UI.shufflingMethod_ComboBox.setEnabled(True);

        self.learning_Setup_UI.trainingPatternMatchingMaking_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatchingDelete_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatchingUp_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatchingDown_Button.setEnabled(True);

        self.learning_Setup_UI.testPatternMatchingMaking_Button.setEnabled(True);
        self.learning_Setup_UI.testPatternMatchingDelete_Button.setEnabled(True);
    
    def Learning_Setup_UI_Training_Viewing_Mode_Widget_Enable(self):
        self.learning_Setup_UI.trainingPatternMatchingMaking_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatchingEnd_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatching_ListWidget.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatchingDelete_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatchingUp_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatchingDown_Button.setEnabled(True);

        self.learning_Setup_UI.trainingPatternPack_ComboBox.setEnabled(False);
        self.learning_Setup_UI.trainingProcess_ComboBox.setEnabled(False);
        self.learning_Setup_UI.trainingAutoAssign_Button.setEnabled(False);

        self.learning_Setup_UI.trainingPattern_ComboBox.setEnabled(False);
        self.learning_Setup_UI.trainingOrder_ComboBox.setEnabled(False);
        self.learning_Setup_UI.trainingPatternToOrderAssign_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternToOrderDelete_Button.setEnabled(False);

        self.learning_Setup_UI.testPatternMatching_GroupBox.setEnabled(True);
        self.learning_Setup_UI.learningSetupEnd_Button.setEnabled(True);

    def Learning_Setup_UI_Training_Matching_Mode_Widget_Enable(self):
        self.learning_Setup_UI.trainingPatternMatchingMaking_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatchingEnd_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternMatching_ListWidget.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatchingDelete_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatchingUp_Button.setEnabled(False);
        self.learning_Setup_UI.trainingPatternMatchingDown_Button.setEnabled(False);

        self.learning_Setup_UI.trainingPatternPack_ComboBox.setEnabled(True);
        self.learning_Setup_UI.trainingProcess_ComboBox.setEnabled(True);
        self.learning_Setup_UI.trainingAutoAssign_Button.setEnabled(True);

        self.learning_Setup_UI.trainingPattern_ComboBox.setEnabled(True);
        self.learning_Setup_UI.trainingOrder_ComboBox.setEnabled(True);
        self.learning_Setup_UI.trainingPatternToOrderAssign_Button.setEnabled(True);
        self.learning_Setup_UI.trainingPatternToOrderDelete_Button.setEnabled(True);

        self.learning_Setup_UI.testPatternMatching_GroupBox.setEnabled(False);
        self.learning_Setup_UI.learningSetupEnd_Button.setEnabled(False);

    def Learning_Setup_UI_Test_Viewing_Mode_Widget_Enable(self):
        self.learning_Setup_UI.testPatternMatchingMaking_Button.setEnabled(True);
        self.learning_Setup_UI.testPatternMatchingEnd_Button.setEnabled(False);
        self.learning_Setup_UI.testPatternMatching_ListWidget.setEnabled(True);
        self.learning_Setup_UI.testPatternMatchingDelete_Button.setEnabled(True);
        
        self.learning_Setup_UI.testPatternPack_ComboBox.setEnabled(False);
        self.learning_Setup_UI.testProcess_ComboBox.setEnabled(False);
        self.learning_Setup_UI.testAutoAssign_Button.setEnabled(False);

        self.learning_Setup_UI.testPattern_ComboBox.setEnabled(False);
        self.learning_Setup_UI.testOrder_ComboBox.setEnabled(False);
        self.learning_Setup_UI.testPatternToOrderAssign_Button.setEnabled(False);
        self.learning_Setup_UI.testPatternToOrderDelete_Button.setEnabled(False);

        self.learning_Setup_UI.extractDataPattern_ComboBox.setEnabled(False);
        self.learning_Setup_UI.extractDataOrder_ComboBox.setEnabled(False);
        self.learning_Setup_UI.extractDataType_ComboBox.setEnabled(False);
        self.learning_Setup_UI.extractDataAssign_Button.setEnabled(False);
        self.learning_Setup_UI.extractDataDelete_Button.setEnabled(False);

        self.learning_Setup_UI.trainingPatternMatching_GroupBox.setEnabled(True);
        self.learning_Setup_UI.learningSetupEnd_Button.setEnabled(True);

    def Learning_Setup_UI_Test_Matching_Mode_Widget_Enable(self):
        self.learning_Setup_UI.testPatternMatchingMaking_Button.setEnabled(False);
        self.learning_Setup_UI.testPatternMatchingEnd_Button.setEnabled(True);
        self.learning_Setup_UI.testPatternMatching_ListWidget.setEnabled(False);
        self.learning_Setup_UI.testPatternMatchingDelete_Button.setEnabled(False);
        
        self.learning_Setup_UI.testPatternPack_ComboBox.setEnabled(True);
        self.learning_Setup_UI.testProcess_ComboBox.setEnabled(True);
        self.learning_Setup_UI.testAutoAssign_Button.setEnabled(True);

        self.learning_Setup_UI.testPattern_ComboBox.setEnabled(True);
        self.learning_Setup_UI.testOrder_ComboBox.setEnabled(True);
        self.learning_Setup_UI.testPatternToOrderAssign_Button.setEnabled(True);
        self.learning_Setup_UI.testPatternToOrderDelete_Button.setEnabled(True);

        self.learning_Setup_UI.extractDataOrder_ComboBox.setEnabled(True);
        self.learning_Setup_UI.extractDataType_ComboBox.setEnabled(True);
        self.learning_Setup_UI.extractDataAssign_Button.setEnabled(True);
        self.learning_Setup_UI.extractDataDelete_Button.setEnabled(True);

        self.learning_Setup_UI.trainingPatternMatching_GroupBox.setEnabled(False);
        self.learning_Setup_UI.learningSetupEnd_Button.setEnabled(False);

    def Learning_Setup_UI_Learning_Setup_Changed(self):
        self.learning_Setup_UI.learningSetupName_LineEdit.setText("");
        self.learning_Setup_UI.learningSetup_ListWidget.clear();

        for learning_Setup in self.simulator.learning_Setup_List:            
            self.learning_Setup_UI.learningSetup_ListWidget.addItem(learning_Setup["Name"]);

        self.Learning_Setup_UI_Current_Learning_Setup_Changed();

    def Learning_Setup_UI_Current_Learning_Setup_Changed(self):
        selected_Index = self.learning_Setup_UI.learningSetup_ListWidget.currentRow();

        self.learning_Setup_UI.trainingEpoch_LineEdit.setText("");
        self.learning_Setup_UI.testTiming_LineEdit.setText("");
        self.learning_Setup_UI.miniBatchSize_LineEdit.setText("");
        self.learning_Setup_UI.shufflingMethod_ComboBox.setCurrentIndex(0);
        self.current_Training_Matching_List = [];
        self.current_Test_Matching_List = [];
        self.learning_Setup_UI.trainingPatternMatching_ListWidget.clear();
        self.learning_Setup_UI. testPatternMatching_ListWidget.clear();

        if selected_Index > -1:
            self.learning_Setup_UI.trainingEpoch_LineEdit.setText(str(self.simulator.learning_Setup_List[selected_Index]["Training_Epoch"]));
            self.learning_Setup_UI.testTiming_LineEdit.setText(str(self.simulator.learning_Setup_List[selected_Index]["Test_Timing"]));
            self.learning_Setup_UI.miniBatchSize_LineEdit.setText(str(self.simulator.learning_Setup_List[selected_Index]["Mini_Batch_Size"]));
            self.learning_Setup_UI.shufflingMethod_ComboBox.setCurrentIndex(self.simulator.learning_Setup_List[selected_Index]["Shuffle_Mode"].value);
            self.current_Training_Matching_List = self.simulator.learning_Setup_List[selected_Index]["Training_Matching_List"];
            self.current_Test_Matching_List = self.simulator.learning_Setup_List[selected_Index]["Test_Matching_List"];

            for training_Matching in self.current_Training_Matching_List:
                self.learning_Setup_UI.trainingPatternMatching_ListWidget.addItem(training_Matching["Pattern_Pack_Name"] + " & " + training_Matching["Process_Name"]);

            for test_Matching in self.current_Test_Matching_List:
                self.learning_Setup_UI.testPatternMatching_ListWidget.addItem(test_Matching["Pattern_Pack_Name"] + " & " + test_Matching["Process_Name"]);

        self.Learning_Setup_UI_Current_Training_Matching_List_Changed();
        self.Learning_Setup_UI_Current_Test_Matching_List_Changed();
    
    def Learning_Setup_UI_Current_Training_Matching_List_Changed(self):
        self.learning_Setup_UI.trainingPatternMatching_ListWidget.clear(); 

        for training_Matching_Information in self.current_Training_Matching_List:
            matching_Information = "PP: " + training_Matching_Information["Pattern_Pack_Name"] + " ↔ Pr: " + training_Matching_Information["Process_Name"];
            self.learning_Setup_UI.trainingPatternMatching_ListWidget.addItem(matching_Information);

        self.Learning_Setup_UI_Current_Training_Matching_Changed();

    def Learning_Setup_UI_Current_Test_Matching_List_Changed(self):
        self.learning_Setup_UI.testPatternMatching_ListWidget.clear(); 

        for test_Matching_Information in self.current_Test_Matching_List:
            matching_Information = "PP: " + test_Matching_Information["Pattern_Pack_Name"] + " ↔ Pr: " + test_Matching_Information["Process_Name"];
            self.learning_Setup_UI.testPatternMatching_ListWidget.addItem(matching_Information);

        self.Learning_Setup_UI_Current_Test_Matching_Changed();
        
    def Learning_Setup_UI_Current_Training_Matching_Changed(self):        
        self.current_Training_Matching_Information = {};
        self.current_Training_Matching_Information["Pattern_Pack_Name"] = "";
        self.current_Training_Matching_Information["Process_Name"] = "";
        self.current_Training_Matching_Information["Assign"] = {};
        self.learning_Setup_UI.trainingPatternPack_ComboBox.setCurrentIndex(0);
        self.learning_Setup_UI.trainingProcess_ComboBox.setCurrentIndex(0);        

        selected_Pattern_Matching_Index = self.learning_Setup_UI.trainingPatternMatching_ListWidget.currentRow();        
        if selected_Pattern_Matching_Index > -1:
            self.learning_Setup_UI.trainingPatternPack_ComboBox.setCurrentText(self.current_Training_Matching_List[selected_Pattern_Matching_Index]["Pattern_Pack_Name"])
            self.learning_Setup_UI.trainingProcess_ComboBox.setCurrentText(self.current_Training_Matching_List[selected_Pattern_Matching_Index]["Process_Name"]);
            self.current_Training_Matching_Information = self.current_Training_Matching_List[selected_Pattern_Matching_Index];

        self.Learning_Setup_UI_Current_Training_Matching_Assign_Changed();
            
    def Learning_Setup_UI_Current_Training_Matching_Pattern_Pack_and_Process_Changed(self):
        self.current_Training_Matching_Information["Pattern_Pack_Name"] = self.learning_Setup_UI.trainingPatternPack_ComboBox.currentText();
        self.current_Training_Matching_Information["Process_Name"] = self.learning_Setup_UI.trainingProcess_ComboBox.currentText();
        self.current_Training_Matching_Information["Assign"] = {};

        self.Learning_Setup_UI_Current_Training_Matching_Assign_Changed();

    def Learning_Setup_UI_Current_Test_Matching_Changed(self):
        self.current_Test_Matching_Information = {};
        self.current_Test_Matching_Information["Pattern_Pack_Name"] = "";
        self.current_Test_Matching_Information["Process_Name"] = "";
        self.current_Test_Matching_Information["Assign"] = {};
        self.current_Test_Matching_Information["Extract_Data"] = [];

        self.learning_Setup_UI.testPatternPack_ComboBox.setCurrentIndex(0);
        self.learning_Setup_UI.testProcess_ComboBox.setCurrentIndex(0);

        selected_Pattern_Matching_Index = self.learning_Setup_UI.testPatternMatching_ListWidget.currentRow();
        if selected_Pattern_Matching_Index > -1:            
            self.learning_Setup_UI.testPatternPack_ComboBox.setCurrentText(self.current_Test_Matching_List[selected_Pattern_Matching_Index]["Pattern_Pack_Name"])
            self.learning_Setup_UI.testProcess_ComboBox.setCurrentText(self.current_Test_Matching_List[selected_Pattern_Matching_Index]["Process_Name"]);
            self.current_Test_Matching_Information = self.current_Test_Matching_List[selected_Pattern_Matching_Index];

        self.Learning_Setup_UI_Current_Test_Matching_Assign_Changed();
        self.Learning_Setup_UI_Current_Test_Extract_Data_Changed();
            
    def Learning_Setup_UI_Current_Test_Matching_Pattern_Pack_and_Process_Changed(self):
        self.current_Test_Matching_Information["Pattern_Pack_Name"] = self.learning_Setup_UI.testPatternPack_ComboBox.currentText();
        self.current_Test_Matching_Information["Process_Name"] = self.learning_Setup_UI.testProcess_ComboBox.currentText();
        self.current_Test_Matching_Information["Assign"] = {};
        self.current_Test_Matching_Information["Extract_Data"] = [];

        self.Learning_Setup_UI_Current_Test_Matching_Assign_Changed();
        self.Learning_Setup_UI_Current_Test_Extract_Data_Changed();

    def Learning_Setup_UI_Current_Training_Matching_Assign_Changed(self):
        self.learning_Setup_UI.trainingPatternToOrderInformation_ListWidget.clear();
        self.learning_Setup_UI.trainingPattern_ComboBox.clear();
        self.learning_Setup_UI.trainingOrder_ComboBox.clear();

        if self.current_Training_Matching_Information["Pattern_Pack_Name"] in self.simulator.pattern_Pack_Dict.keys() and self.current_Training_Matching_Information["Process_Name"] in self.simulator.process_Dict.keys():
            selected_Pattern_Pack = self.simulator.pattern_Pack_Dict[self.current_Training_Matching_Information["Pattern_Pack_Name"]];
            selected_Process = self.simulator.process_Dict[self.current_Training_Matching_Information["Process_Name"]];

            display_Assign_List = [];
            for order_Index in self.current_Training_Matching_Information["Assign"].keys():                        
                if selected_Process["Order_List"][order_Index][0] == Order_Code.Input_Layer_Acitvation_Insert:
                    display_Assign_List.append(str(order_Index) + ": Act. In ← " + self.current_Training_Matching_Information["Assign"][order_Index]);
                elif selected_Process["Order_List"][order_Index][0] == Order_Code.Output_Layer_Error_Calculation_Sigmoid:
                    display_Assign_List.append(str(order_Index) + ": Error Calc. Sigmoid ← " + self.current_Training_Matching_Information["Assign"][order_Index]);
                elif selected_Process["Order_List"][order_Index][0] == Order_Code.Output_Layer_Error_Calculation_Softmax:
                    display_Assign_List.append(str(order_Index) + ": Error Calc. Softmax ← " + self.current_Training_Matching_Information["Assign"][order_Index]);                
            for display_Line in display_Assign_List:
                self.learning_Setup_UI.trainingPatternToOrderInformation_ListWidget.addItem(display_Line);        
        
            for pattern_Name in selected_Pattern_Pack.keys():
                if pattern_Name in ["Name", "Probability", "Cycle", "Count"]:
                    continue;
                self.learning_Setup_UI.trainingPattern_ComboBox.addItem(pattern_Name);

            for order_Index in range(len(selected_Process["Order_List"])):            
                order_Code, layer_List, connection_List, order_Variable_List = selected_Process["Order_List"][order_Index];
                if order_Index in self.current_Training_Matching_Information["Assign"].keys():
                    continue;
                elif order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax]:
                    self.learning_Setup_UI.trainingOrder_ComboBox.addItem(str(order_Index) + ": " + str(order_Code)[11:] + "(" + layer_List[0] + ")");

            if self.learning_Setup_UI.trainingOrder_ComboBox.count() == 0:
                self.learning_Setup_UI.trainingPatternToOrderAssign_Button.setEnabled(False);
            else:
                self.learning_Setup_UI.trainingPatternToOrderAssign_Button.setEnabled(True);
        
    def Learning_Setup_UI_Current_Test_Matching_Assign_Changed(self):
        self.learning_Setup_UI.testPatternToOrderInformation_ListWidget.clear();
        self.learning_Setup_UI.testPattern_ComboBox.clear();
        self.learning_Setup_UI.testOrder_ComboBox.clear();

        if self.current_Test_Matching_Information["Pattern_Pack_Name"] in self.simulator.pattern_Pack_Dict.keys() and self.current_Test_Matching_Information["Process_Name"] in self.simulator.process_Dict.keys():
            selected_Pattern_Pack = self.simulator.pattern_Pack_Dict[self.current_Test_Matching_Information["Pattern_Pack_Name"]];
            selected_Process = self.simulator.process_Dict[self.current_Test_Matching_Information["Process_Name"]];

            display_Assign_List = [];
            for order_Index in self.current_Test_Matching_Information["Assign"].keys():                        
                if selected_Process["Order_List"][order_Index][0] == Order_Code.Input_Layer_Acitvation_Insert:
                    display_Assign_List.append(str(order_Index) + ": Act. In ← " + self.current_Test_Matching_Information["Assign"][order_Index]);
                elif selected_Process["Order_List"][order_Index][0] == Order_Code.Output_Layer_Error_Calculation_Sigmoid:
                    display_Assign_List.append(str(order_Index) + ": Error Calc. Sigmoid ← " + self.current_Test_Matching_Information["Assign"][order_Index]);
                elif selected_Process["Order_List"][order_Index][0] == Order_Code.Output_Layer_Error_Calculation_Softmax:
                    display_Assign_List.append(str(order_Index) + ": Error Calc. Softmax ← " + self.current_Test_Matching_Information["Assign"][order_Index]);                
            for display_Line in display_Assign_List:
                self.learning_Setup_UI.testPatternToOrderInformation_ListWidget.addItem(display_Line);        
        
            for pattern_Name in selected_Pattern_Pack.keys():
                if pattern_Name in ["Name", "Probability", "Cycle", "Count"]:
                    continue;
                self.learning_Setup_UI.testPattern_ComboBox.addItem(pattern_Name);

            for order_Index in range(len(selected_Process["Order_List"])):            
                order_Code, layer_List, connection_List, order_Variable_List = selected_Process["Order_List"][order_Index];
                if order_Index in self.current_Test_Matching_Information["Assign"].keys():
                    continue;
                elif order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax]:
                    self.learning_Setup_UI.testOrder_ComboBox.addItem(str(order_Index) + ": " + str(order_Code)[11:] + "(" + layer_List[0] + ")");

            if self.learning_Setup_UI.testOrder_ComboBox.count() == 0:
                self.learning_Setup_UI.testPatternToOrderAssign_Button.setEnabled(False);
            else:
                self.learning_Setup_UI.testPatternToOrderAssign_Button.setEnabled(True);

    def Learning_Setup_UI_Current_Test_Extract_Data_Changed(self):
        self.learning_Setup_UI.extractData_ListWidget.clear();
        self.learning_Setup_UI.extractDataPattern_ComboBox.clear();
        self.learning_Setup_UI.extractDataOrder_ComboBox.clear();
        self.learning_Setup_UI.extractDataType_ComboBox.setCurrentIndex(0);

        if self.current_Test_Matching_Information["Pattern_Pack_Name"] in self.simulator.pattern_Pack_Dict.keys() and self.current_Test_Matching_Information["Process_Name"] in self.simulator.process_Dict.keys():
            selected_Pattern_Pack = self.simulator.pattern_Pack_Dict[self.current_Test_Matching_Information["Pattern_Pack_Name"]];
            selected_Process = self.simulator.process_Dict[self.current_Test_Matching_Information["Process_Name"]];

            display_Extract_List = [];
            for pattern, order_Index, extract_Type in self.current_Test_Matching_Information["Extract_Data"]:
                extract_Layer_Name = selected_Process["Order_List"][order_Index][1][0];                
                if extract_Type == Extract_Data_Type.Raw_Activation:
                    display_Extract_List.append("Raw Activation(L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
                elif extract_Type == Extract_Data_Type.Mean_Squared_Error:
                    display_Extract_List.append("Mean Squared Error(T: " + pattern + ", L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
                elif extract_Type == Extract_Data_Type.Cross_Entropy:
                    display_Extract_List.append("Cross Entropy(T: " + pattern + ", L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
                elif extract_Type == Extract_Data_Type.Semantic_Stress:
                    display_Extract_List.append("Semantic Stress(L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
            for display_Line in display_Extract_List:
                self.learning_Setup_UI.extractData_ListWidget.addItem(display_Line);

            for pattern_Name in selected_Pattern_Pack.keys():
                if pattern_Name in ["Name", "Probability", "Cycle", "Count"]:
                    continue;
                self.learning_Setup_UI.extractDataPattern_ComboBox.addItem(pattern_Name);

            for order_Index in range(len(selected_Process["Order_List"])):            
                order_Code, layer_List, connection_List, order_Variable_List = selected_Process["Order_List"][order_Index];                
                if order_Code == Order_Code.Activation_Extract:
                    self.learning_Setup_UI.extractDataOrder_ComboBox.addItem(str(order_Index) + ": " + layer_List[0]);
    # End Learning Setup Functions
    
    # Start Learning Functions
    def Learning_UI_start_Button_Clicked(self):
        self.simulator.pause_Status = False;
        self.learning_UI.start_Button.setEnabled(False);
        self.learning_UI.testResultSave_Button.setEnabled(False);
        self.learning_UI.weightAndBiasSave_Button.setEnabled(False);
        self.learning_UI.exit_Button.setEnabled(False);
        self.learning_UI.pause_Button.setEnabled(True);

        thread.start_new_thread(self.simulator.Learn, ());
        thread.start_new_thread(self.Learning_UI_Simulator_Learning_Display, ());                    
        thread.start_new_thread(self.Learning_UI_Simulator_Weight_Graph_Display, ());
        thread.start_new_thread(self.Learning_UI_Simulator_Result_Graph_Display, ());

    def Learning_UI_pause_Button_Clicked(self):
        self.simulator.pause_Status = True;
        self.learning_UI.start_Button.setEnabled(True);
        self.learning_UI.testResultSave_Button.setEnabled(True);
        self.learning_UI.weightAndBiasSave_Button.setEnabled(True);
        self.learning_UI.exit_Button.setEnabled(True);
        self.learning_UI.pause_Button.setEnabled(False);

    def Learning_UI_testResultSave_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getExistingDirectory()
        if file_Path != "":                    
            self.simulator.Test_Result_Save(file_Path);

    def Learning_UI_weightAndBiasSave_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getSaveFileName(filter="Model File for HNet (*.HNet_Model)")[0]
        if file_Path != "":
            self.simulator.WeightAndBias_Save(file_Path);

    def Learning_UI_weight_Display_Button_Clicked(self):
        try:
            key = self.learning_UI.weightName_ComboBox.currentText().strip().split(" ");
            self.learning_UI.weight_Graph.Update_Figure("From Layer", "To Layer", self.simulator.weight_Dict_for_Observation[(key[0], key[1])]);
        except:
            pass;

    def Learning_UI_result_Display_Button_Clicked(self):
        yAxis = [-0.01, 1.01];  #Prevent Error
        #While yAxis' line_Edit modify  
        try: yAxis = [float(self.learning_UI.yAxisMin_LineEdit.text()), float(self.learning_UI.yAxisMax_LineEdit.text())];
        except: pass;

        if not self.learning_UI.cycle_CheckBox.isChecked():
            display_Mode_Index = self.learning_UI.displayMode_ComboBox.currentIndex();
            if display_Mode_Index == 0:
                xTicks, averaged_Result_Array = self.Learning_UI_Simulator_Extract_Mean_Squared_Error_for_Graph();
            if display_Mode_Index == 1:
                xTicks, averaged_Result_Array = self.Learning_UI_Simulator_Extract_Cross_Entropy_for_Graph();
            if display_Mode_Index == 2:
                xTicks, averaged_Result_Array = self.Learning_UI_Simulator_Extract_Semantic_Stress_for_Graph();
            self.learning_UI.result_Graph.Update_Figure(xTicks, averaged_Result_Array, yAxis);
        else:
            display_Mode_Index = self.learning_UI.displayMode_ComboBox.currentIndex();
            if display_Mode_Index == 0:
                max_Cycle, xTicks, averaged_Result_Array = self.Learning_UI_Simulator_Extract_Mean_Squared_Error_by_Cycle_for_Graph();
            if display_Mode_Index == 1:
                max_Cycle, xTicks, averaged_Result_Array = self.Learning_UI_Simulator_Extract_Cross_Entropy_by_Cycle_for_Graph();
            if display_Mode_Index == 2:
                max_Cycle, xTicks, averaged_Result_Array = self.Learning_UI_Simulator_Extract_Semantic_Stress_by_Cycle_for_Graph();
            self.learning_UI.result_Graph.Update_Figure_using_Cycle(xTicks, averaged_Result_Array, yAxis, max_Cycle);

    def Learning_UI_exit_Button_Clicked(self):
        self.learning_Dialog.hide();
        self.main_Dialog.exec_();
    
    def Learning_UI_ComboBox_Add_Item(self):
        self.learning_UI.weightName_ComboBox.clear();
        self.learning_UI.weightName_ComboBox.addItem("");
        for connection in self.simulator.connection_Information_Dict.keys():
            self.learning_UI.weightName_ComboBox.addItem("Weight " + connection);
        for layer in self.simulator.layer_Information_Dict.keys():
            self.learning_UI.weightName_ComboBox.addItem("Bias " + layer);

    def Learning_UI_Simulator_Learning_Display(self):
        while not self.simulator.pause_Status:
            time.sleep(0.01);
            self.learning_UI.totalEpoch_LineEdit.setText(str(self.simulator.current_Total_Epoch));
            if self.simulator.current_Learning_Setup_Index < len(self.simulator.learning_Setup_List):                    
                self.learning_UI.currentLearningSetup_LineEdit.setText(str(self.simulator.learning_Setup_List[self.simulator.current_Learning_Setup_Index]["Name"]));
            self.learning_UI.currentEpoch_LineEdit.setText(str(self.simulator.current_LearningSetup_Epoch));
        self.learning_UI.totalEpoch_LineEdit.setText(str(self.simulator.current_Total_Epoch));

        self.Learning_UI_pause_Button_Clicked();

    def Learning_UI_Simulator_Weight_Graph_Display(self):
        while not self.simulator.pause_Status:
            time.sleep(0.1);            
            try:    #Macro conflict                
                self.Learning_UI_weight_Display_Button_Clicked();
            except:
                continue;

    def Learning_UI_Simulator_Result_Graph_Display(self):
        while not self.simulator.pause_Status:
            time.sleep(1.0);            
            try:    #Macro conflict
                self.Learning_UI_result_Display_Button_Clicked();
            except:
                continue;

    def Learning_UI_Simulator_Extract_Mean_Squared_Error_for_Graph(self):
        raw_Mean_Squared_Error_Dict = {};
        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.simulator.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Mean_Squared_Error:
                continue; 
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.simulator.extract_Result_Dict[data_Key];
            target_Pattern = self.simulator.pattern_Pack_Dict[pattern_Pack_Name][pattern_Name];
                        
            if not total_Epoch in raw_Mean_Squared_Error_Dict.keys():
                #The reason which data is list: At same epoch, multi test can be occurred.
                raw_Mean_Squared_Error_Dict[total_Epoch] = [];
            raw_Mean_Squared_Error_Dict[total_Epoch].append(np.sqrt(np.mean((target_Pattern - raw_Data) ** 2)));
        
        xTick_List = self.Learning_UI_Simulator_Extract_All_Test_Timing_for_Graph();
        
        averaged_Result_List = [];
        for epoch in xTick_List:
            if epoch in raw_Mean_Squared_Error_Dict.keys():
                averaged_Result_List.append(np.mean(raw_Mean_Squared_Error_Dict[epoch]));
            else:
                averaged_Result_List.append(None);

        return np.array(xTick_List), np.array(averaged_Result_List);        

    def Learning_UI_Simulator_Extract_Cross_Entropy_for_Graph(self):
        raw_Cross_Entropy_Dict = {};
        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.simulator.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Cross_Entropy:
                continue;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.simulator.extract_Result_Dict[data_Key];
            target_Pattern = self.simulator.pattern_Pack_Dict[pattern_Pack_Name][pattern_Name];
            
            if not total_Epoch in raw_Cross_Entropy_Dict.keys():
                raw_Cross_Entropy_Dict[total_Epoch] = [];
            raw_Cross_Entropy_Dict[total_Epoch].append(-(np.mean(target_Pattern * np.log(raw_Data) + (1 - target_Pattern) * np.log(1 - raw_Data))));

        xTick_List = self.Learning_UI_Simulator_Extract_All_Test_Timing_for_Graph();
        
        averaged_Result_List = [];
        for epoch in xTick_List:
            if epoch in raw_Cross_Entropy_Dict.keys():
                averaged_Result_List.append(np.mean(raw_Cross_Entropy_Dict[epoch]));
            else:
                averaged_Result_List.append(None);

        return np.array(xTick_List), np.array(averaged_Result_List);       

    def Learning_UI_Simulator_Extract_Semantic_Stress_for_Graph(self):
        raw_Semantic_Stress_Dict = {};
        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.simulator.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Semantic_Stress:
                continue;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.simulator.extract_Result_Dict[data_Key];
            
            if not total_Epoch in raw_Semantic_Stress_Dict.keys():
                raw_Semantic_Stress_Dict[total_Epoch] = [];
            raw_Semantic_Stress_Dict[total_Epoch].append(np.mean(raw_Data * np.log2(raw_Data) + (1-raw_Data) * np.log2(1-raw_Data) + 1));
            
        xTick_List = self.Learning_UI_Simulator_Extract_All_Test_Timing_for_Graph();
        
        averaged_Result_List = [];
        for epoch in xTick_List:
            if epoch in raw_Semantic_Stress_Dict.keys():
                averaged_Result_List.append(np.mean(raw_Semantic_Stress_Dict[epoch]));
            else:
                averaged_Result_List.append(None);

        return np.array(xTick_List), np.array(averaged_Result_List);

    def Learning_UI_Simulator_Extract_Mean_Squared_Error_by_Cycle_for_Graph(self):
        max_Cycle = 0;
        raw_Mean_Squared_Error_Dict = {};        
        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.simulator.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Mean_Squared_Error:
                continue; 
            if max_Cycle < cycle:
                max_Cycle = cycle;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.simulator.extract_Result_Dict[data_Key];
            target_Pattern = self.simulator.pattern_Pack_Dict[pattern_Pack_Name][pattern_Name];
            
            if not total_Epoch in raw_Mean_Squared_Error_Dict.keys():
                #The reason which data is list: At same epoch, multi test can be occurred.
                raw_Mean_Squared_Error_Dict[(total_Epoch, cycle)] = [];
            raw_Mean_Squared_Error_Dict[(total_Epoch, cycle)].append(np.sqrt(np.mean((target_Pattern - raw_Data) ** 2)));
        
        xTick_List = self.Learning_UI_Simulator_Extract_All_Test_Timing_for_Graph();
        
        epoch_Row_Index_Dict = {};
        current_Row_Index = 0;
        for epoch in xTick_List:
            epoch_Row_Index_Dict[epoch] = current_Row_Index;
            current_Row_Index += 1;

        averaged_Result_Array = np.zeros(shape=(len(xTick_List), max_Cycle + 1));
        for epoch in xTick_List:
            for cycle in range(max_Cycle + 1):
                if (epoch, cycle) in raw_Mean_Squared_Error_Dict.keys():
                    averaged_Result_Array[epoch_Row_Index_Dict[epoch], cycle] = np.mean(raw_Mean_Squared_Error_Dict[(epoch, cycle)]);
                else:
                    averaged_Result_Array[epoch_Row_Index_Dict[epoch], cycle] = None;

        return max_Cycle, np.array(xTick_List), averaged_Result_Array;        

    def Learning_UI_Simulator_Extract_Cross_Entropy_by_Cycle_for_Graph(self):
        max_Cycle = 0;
        raw_Cross_Entropy_Dict = {};
        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.simulator.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Cross_Entropy:
                continue;
            if max_Cycle < cycle:
                max_Cycle = cycle;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.simulator.extract_Result_Dict[data_Key];
            target_Pattern = self.simulator.pattern_Pack_Dict[pattern_Pack_Name][pattern_Name];
            
            if not total_Epoch in raw_Cross_Entropy_Dict.keys():
                raw_Cross_Entropy_Dict[(total_Epoch, cycle)] = [];
            raw_Cross_Entropy_Dict[(total_Epoch, cycle)].append(-(np.mean(target_Pattern * np.log(raw_Data) + (1 - target_Pattern) * np.log(1 - raw_Data))));

        xTick_List = self.Learning_UI_Simulator_Extract_All_Test_Timing_for_Graph();
        
        epoch_Row_Index_Dict = {};
        current_Row_Index = 0;
        for epoch in xTick_List:
            epoch_Row_Index_Dict[epoch] = current_Row_Index;
            current_Row_Index += 1;

        averaged_Result_Array = np.zeros(shape=(len(xTick_List), max_Cycle + 1));
        for epoch in xTick_List:
            for cycle in range(max_Cycle + 1):
                if (epoch, cycle) in raw_Cross_Entropy_Dict.keys():
                    averaged_Result_Array[epoch_Row_Index_Dict[epoch], cycle] = np.mean(raw_Cross_Entropy_Dict[(epoch, cycle)]);
                else:
                    averaged_Result_Array[epoch_Row_Index_Dict[epoch], cycle] = None;

        return max_Cycle, np.array(xTick_List), averaged_Result_Array;        

    def Learning_UI_Simulator_Extract_Semantic_Stress_by_Cycle_for_Graph(self):
        max_Cycle = 0;
        raw_Semantic_Stress_Dict = {};
        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.simulator.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Semantic_Stress:
                continue;
            if max_Cycle < cycle:
                max_Cycle = cycle;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.simulator.extract_Result_Dict[data_Key];
            
            if not total_Epoch in raw_Semantic_Stress_Dict.keys():
                raw_Semantic_Stress_Dict[(total_Epoch, cycle)] = [];
            raw_Semantic_Stress_Dict[(total_Epoch, cycle)].append(np.mean(raw_Data * np.log2(raw_Data) + (1-raw_Data) * np.log2(1-raw_Data) + 1));
            
        xTick_List = self.Learning_UI_Simulator_Extract_All_Test_Timing_for_Graph();
        
        epoch_Row_Index_Dict = {};
        current_Row_Index = 0;
        for epoch in xTick_List:
            epoch_Row_Index_Dict[epoch] = current_Row_Index;
            current_Row_Index += 1;

        averaged_Result_Array = np.zeros(shape=(len(xTick_List), max_Cycle + 1));
        for epoch in xTick_List:
            for cycle in range(max_Cycle + 1):
                if (epoch, cycle) in raw_Semantic_Stress_Dict.keys():
                    averaged_Result_Array[epoch_Row_Index_Dict[epoch], cycle] = np.mean(raw_Semantic_Stress_Dict[(epoch, cycle)]);
                else:
                    averaged_Result_Array[epoch_Row_Index_Dict[epoch], cycle] = None;

        return max_Cycle, np.array(xTick_List), averaged_Result_Array;

    def Learning_UI_Simulator_Extract_All_Test_Timing_for_Graph(self):
        xTick_List = [];
        total_Epoch = 0;
        for learning_Setup in self.simulator.learning_Setup_List:
            timing_List = np.arange(0, learning_Setup["Training_Epoch"] + 1, learning_Setup["Test_Timing"]) + total_Epoch;
            xTick_List.extend(timing_List);
            total_Epoch += learning_Setup["Training_Epoch"];

        xTick_List = list(set(xTick_List)); #When a learning Setup is changed, epoch will be overlapped.
        xTick_List.sort();

        return xTick_List; 
            
    # End Learning Functions
    
    # Start Macro Functions
    #Add function problem
    def Macro_UI_baseModelDataFileBroswer_Button_Clicked(self):
        new_FileDialog = QtWidgets.QFileDialog();
        file_Path = new_FileDialog.getOpenFileName(filter="Structure File for HNet (*.HNetGUI_Model_Data_for_Macro)")[0]        
        if file_Path == "":
            return;

        self.macro_UI.baseModelDataFile_LineEdit.setText(file_Path);

    def Macro_UI_baseModelDataFileLoad_Button_Clicked(self):
        file_Path = self.macro_UI.baseModelDataFile_LineEdit.text();

        if file_Path == "":
            return;

        with open(file_Path, "rb") as f:
            self.macro_Base_Dict = pickle.load(f);
        
        self.Macro_UI_base_Model_Changed();
    
    def Macro_UI_macroAdd_Button_Clicked(self):
        modified_Dict_List = [self.macro_Base_Dict];
        for modify_Factor in self.current_Modify_Factor_List:
            modified_Dict_List = self.Macro_UI_Modifying_Factor_Apply(modified_Dict_List, modify_Factor);

        for modified_Dict in modified_Dict_List:
            new_Simulator = HNet();
            new_Simulator.config_Variables_Dict = modified_Dict["Config_Dict"];
            new_Simulator.layer_Information_Dict = modified_Dict["Layer_Dict"];
            new_Simulator.connection_Information_Dict = modified_Dict["Connection_Dict"];
            new_Simulator.pattern_Pack_Dict = modified_Dict["Pattern_Pack_Dict"];
            new_Simulator.process_Dict = modified_Dict["Process_Dict"];
            new_Simulator.learning_Setup_List = modified_Dict["Learning_Setup_List"];
            self.modified_Simulator_List.append(new_Simulator);

        self.current_Modify_Factor_List = [];        
        self.Macro_UI_Modified_Simulator_Changed();
        self.Macro_UI_Modify_Factor_Changed();

    def Macro_UI_macroDelete_Button_Clicked(self):
        selected_Index = self.macro_UI.macro_ListWidget.currentRow();

        if selected_Index < 0:
            return;

        del self.modified_Simulator_List[selected_Index];

        self.Macro_UI_Modified_Simulator_Changed();

    def Macro_UI_macro_ListWidget_Current_Item_Changed(self):
        self.macro_UI.macroInformation_TextEdit.setText("");
        select_Index = self.macro_UI.macro_ListWidget.currentRow();
        if select_Index < 0:
            return;

        self.macro_UI.macroInformation_TextEdit.setText(self.modified_Simulator_List[select_Index].Extract_Simulator_Information());
    
    def Macro_UI_modifyFactorDelete_Button_Clicked(self):
        selected_Index = self.macro_UI.modifyFactor_ListWidget.currentRow();
        if selected_Index < 0:
            return;

        del self.current_Modify_Factor_List[selected_Index];

        self.Macro_UI_Modify_Factor_Changed();

    def Macro_UI_layerSizeAdd_Button_Clicked(self):
        if self.macro_UI.layerSizeFrom_LineEdit.text() == "":
            self.macro_UI.layerSizeFrom_LineEdit.setFocus();
            return;
        elif self.macro_UI.layerSizeTo_LineEdit.text() == "":
            self.macro_UI.layerSizeTo_LineEdit.setFocus();
            return;
        elif self.macro_UI.layerSizeStep_LineEdit.text() == "":
            self.macro_UI.layerSizeStep_LineEdit.setFocus();
            return;

        selected_Layer = self.macro_UI.layerSizeLayer_ComboBox.currentText();        
        for process_Key in self.macro_Base_Dict["Process_Dict"].keys():
            process = self.macro_Base_Dict["Process_Dict"][process_Key];
            for order_Code, layer_Name_List, connection_Name_List, order_Variable_List in process["Order_List"]:
                if order_Code in [Order_Code.Connection_Duplication, Order_Code.Transposed_Connection_Duplication, Order_Code.Weight_Equalization]:
                    for connection_Name in connection_Name_List:
                        connection = self.macro_Base_Dict["Connection_Dict"][connection_Name];
                        if selected_Layer == connection["From_Layer_Name"] or selected_Layer == connection["To_Layer_Name"]:
                            QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer which used at a weight control order (Weight Equalization or Duplication) in a process cannot be modified. Try to use 'Multi layer size'.");
                            return;
                elif layer_Name_List is None:
                    continue;
                elif order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax, Order_Code.Activation_Extract] and selected_Layer in layer_Name_List:
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer which used with some pattern in a process cannot be modified.");
                    return;
                elif order_Code == Order_Code.Layer_Duplication and selected_Layer in layer_Name_List:
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer which used at 'Layer Duplication' in a process cannot be modified. Try to use 'Multi layer size'.");
                    return;                
                elif order_Code == Order_Code.Bias_Equalization and selected_Layer in layer_Name_List:
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer which used at 'Bias Equalization' in a process cannot be modified. Try to use 'Multi layer size'.");
                    return;

        for modify_Type, factor, variable_List in self.current_Modify_Factor_List:
            if modify_Type == "Layer_Size" and factor == selected_Layer:
                QtWidgets.QMessageBox.warning(None, 'Warning!', "This layer is already in the list as a modify factor.");
                return;

        modified_Variable_List = [];
        modified_Variable_List.extend(np.arange(int(self.macro_UI.layerSizeFrom_LineEdit.text()), int(self.macro_UI.layerSizeTo_LineEdit.text()), int(self.macro_UI.layerSizeStep_LineEdit.text())));
        modified_Variable_List.append(int(self.macro_UI.layerSizeTo_LineEdit.text()));

        self.current_Modify_Factor_List.append(("Layer_Size", selected_Layer, modified_Variable_List));

        self.Macro_UI_Modify_Factor_Changed();

    def Macro_UI_regularMultiLayerSizeAdd_Button_Clicked(self):
        if self.macro_UI.regularMultiLayerSizeLayerPrefix_LineEdit.text() == "":
            self.macro_UI.regularMultiLayerSizeLayerPrefix_LineEdit.setFocus();
            return;
        elif self.macro_UI.regularMultiLayerSizeMaxSuffix_LineEdit.text() == "":
            self.macro_UI.regularMultiLayerSizeMaxSuffix_LineEdit.setFocus();
            return;
        elif self.macro_UI.regularMultiLayerSizeFrom_LineEdit.text() == "":
            self.macro_UI.layerSizeFrom_LineEdit.setFocus();
            return;
        elif self.macro_UI.regularMultiLayerSizeTo_LineEdit.text() == "":
            self.macro_UI.layerSizeTo_LineEdit.setFocus();
            return;
        elif self.macro_UI.regularMultiLayerSizeStep_LineEdit.text() == "":
            self.macro_UI.layerSizeStep_LineEdit.setFocus();
            return;

        layer_Regular_Name = self.macro_UI.regularMultiLayerSizeLayerPrefix_LineEdit.text();
        max_MaxSuffix = int(self.macro_UI.regularMultiLayerSizeMaxSuffix_LineEdit.text());

        selected_Layer_Name_List = [];        
        for maxSuffix in range(1, max_MaxSuffix + 1):
            selected_Layer_Name_List.append(layer_Regular_Name + str(maxSuffix));

        if set(selected_Layer_Name_List) & set(self.macro_Base_Dict["Layer_Dict"].keys()) != set(selected_Layer_Name_List):
            QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no layer. Check the prefix and max suffix you inserted.");
            return;

        for process_Key in self.macro_Base_Dict["Process_Dict"].keys():
            process = self.macro_Base_Dict["Process_Dict"][process_Key];
            for order_Code, layer_Name_List, connection_Name_List, order_Variable_List in process["Order_List"]:
                if order_Code in [Order_Code.Connection_Duplication, Order_Code.Transposed_Connection_Duplication, Order_Code.Weight_Equalization]:
                    from_Layer_Set = set();
                    to_Layer_Set = set();
                    for connection_Name in connection_Name_List:
                        connection = self.macro_Base_Dict["Connection_Dict"][connection_Name];
                        from_Layer_Set.add(connection["From_Layer_Name"]);
                        to_Layer_Set.add(connection["To_Layer_Name"]);                    
                    if not (set(selected_Layer_Name_List) & from_Layer_Set) in [set(), from_Layer_Set] or not (set(selected_Layer_Name_List) & to_Layer_Set) in [set(), to_Layer_Set]:
                        QtWidgets.QMessageBox.warning(None, 'Warning!', "There is a layer which is used at a weight control order (Weight Equalization or Duplication), but not controlled.");
                        return;
                elif layer_Name_List is None:
                    continue;
                elif order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax, Order_Code.Activation_Extract] and set() != set(selected_Layer_Name_List) & set(layer_Name_List):
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer which used with some pattern in a process cannot be modified.");
                    return;
                elif order_Code == Order_Code.Layer_Duplication and not (set(selected_Layer_Name_List) & set(layer_Name_List)) in [set(), set(layer_Name_List)]:
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "There is a layer which is used at the 'Layer Duplication', but not controlled.");
                    return;
                elif order_Code == Order_Code.Bias_Equalization and not (set(selected_Layer_Name_List) & set(layer_Name_List)) in [set(), set(layer_Name_List)]:
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "There is a layer which is used at the 'Bias Equalization', but not controlled.");
                    return;

        for modify_Type, factor, variable_List in self.current_Modify_Factor_List:
            if (modify_Type == "Layer_Size" and factor in selected_Layer_Name_List) or (modify_Type == "Multi_Layer_Size" and len(set(factor) & set(selected_Layer_Name_List)) > 0):
                QtWidgets.QMessageBox.warning(None, 'Warning!', "A layer is already in the list as a modify factor.");
                return;

        modified_Variable_List = [];
        modified_Variable_List.extend(np.arange(int(self.macro_UI.regularMultiLayerSizeFrom_LineEdit.text()), int(self.macro_UI.regularMultiLayerSizeTo_LineEdit.text()), int(self.macro_UI.regularMultiLayerSizeStep_LineEdit.text())));
        modified_Variable_List.append(int(self.macro_UI.regularMultiLayerSizeTo_LineEdit.text()));

        self.current_Modify_Factor_List.append(("Multi_Layer_Size", selected_Layer_Name_List, modified_Variable_List));

        self.Macro_UI_Modify_Factor_Changed();

    def Macro_UI_irregularMultiLayerSizeAdd_Button_Clicked(self):
        if self.macro_UI.irregularMultiLayerSizeLayer_LineEdit.text() == "":
            self.macro_UI.regularMultiLayerSizeLayerPrefix_LineEdit.setFocus();
            return;
        elif self.macro_UI.irregularMultiLayerSizeFrom_LineEdit.text() == "":
            self.macro_UI.layerSizeFrom_LineEdit.setFocus();
            return;
        elif self.macro_UI.irregularMultiLayerSizeTo_LineEdit.text() == "":
            self.macro_UI.layerSizeTo_LineEdit.setFocus();
            return;
        elif self.macro_UI.irregularMultiLayerSizeStep_LineEdit.text() == "":
            self.macro_UI.layerSizeStep_LineEdit.setFocus();
            return;

        selected_Layer_Name_List = [];        
        selected_Layer_Name_List = self.macro_UI.irregularMultiLayerSizeLayer_LineEdit.text().strip().split(" ");

        if set(selected_Layer_Name_List) & set(self.macro_Base_Dict["Layer_Dict"].keys()) != set(selected_Layer_Name_List):
            QtWidgets.QMessageBox.warning(None, 'Warning!', "There is no layer. Check the layers you inserted.");
            return;

        for process_Key in self.macro_Base_Dict["Process_Dict"].keys():
            process = self.macro_Base_Dict["Process_Dict"][process_Key];
            for order_Code, layer_Name_List, connection_Name_List, order_Variable_List in process["Order_List"]:
                if order_Code in [Order_Code.Connection_Duplication, Order_Code.Transposed_Connection_Duplication, Order_Code.Weight_Equalization]:
                    from_Layer_Set = set();
                    to_Layer_Set = set();
                    for connection_Name in connection_Name_List:
                        connection = self.macro_Base_Dict["Connection_Dict"][connection_Name];
                        from_Layer_Set.add(connection["From_Layer_Name"]);
                        to_Layer_Set.add(connection["To_Layer_Name"]);                    
                    if not (set(selected_Layer_Name_List) & from_Layer_Set) in [set(), from_Layer_Set] or not (set(selected_Layer_Name_List) & to_Layer_Set) in [set(), to_Layer_Set]:
                        QtWidgets.QMessageBox.warning(None, 'Warning!', "There is a layer which is used at a weight control order (Weight Equalization or Duplication), but not controlled.");
                        return;
                elif layer_Name_List is None:
                    continue;
                elif order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax, Order_Code.Activation_Extract] and set() != set(selected_Layer_Name_List) & set(layer_Name_List):
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "The layer which used with some pattern in a process cannot be modified.");
                    return;
                elif order_Code == Order_Code.Layer_Duplication and not (set(selected_Layer_Name_List) & set(layer_Name_List)) in [set(), set(layer_Name_List)]:
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "There is a layer which is used at the 'Layer Duplication', but not controlled.");
                    return;
                elif order_Code == Order_Code.Bias_Equalization and not (set(selected_Layer_Name_List) & set(layer_Name_List)) in [set(), set(layer_Name_List)]:
                    QtWidgets.QMessageBox.warning(None, 'Warning!', "There is a layer which is used at the 'Bias Equalization', but not controlled.");
                    return;

        for modify_Type, factor, variable_List in self.current_Modify_Factor_List:
            if (modify_Type == "Layer_Size" and factor in selected_Layer_Name_List) or (modify_Type == "Multi_Layer_Size" and len(set(factor) & set(selected_Layer_Name_List)) > 0):
                QtWidgets.QMessageBox.warning(None, 'Warning!', "A layer is already in the list as a modify factor.");
                return;

        modified_Variable_List = [];
        modified_Variable_List.extend(np.arange(int(self.macro_UI.irregularMultiLayerSizeFrom_LineEdit.text()), int(self.macro_UI.irregularMultiLayerSizeTo_LineEdit.text()), int(self.macro_UI.irregularMultiLayerSizeStep_LineEdit.text())));
        modified_Variable_List.append(int(self.macro_UI.irregularMultiLayerSizeTo_LineEdit.text()));

        self.current_Modify_Factor_List.append(("Multi_Layer_Size", selected_Layer_Name_List, modified_Variable_List));

        self.Macro_UI_Modify_Factor_Changed();

    def Macro_UI_learningRateAdd_Button_Clicked(self):
        if self.macro_UI.learningRateFrom_LineEdit.text() == "":
            self.macro_UI.learningRateFrom_LineEdit.setFocus();
            return;
        elif self.macro_UI.learningRateTo_LineEdit.text() == "":
            self.macro_UI.learningRateTo_LineEdit.setFocus();
            return;
        elif self.macro_UI.learningRateStep_LineEdit.text() == "":
            self.macro_UI.learningRateStep_LineEdit.setFocus();
            return;

        for modify_Type, factor, variable_List in self.current_Modify_Factor_List:
            if modify_Type == "Learning_Rate":
                QtWidgets.QMessageBox.warning(None, 'Warning!', "Learning rate is already in the list as a modify factor.");
                return;

        modified_Variable_List = [];
        modified_Variable_List.extend(np.arange(float(self.macro_UI.learningRateFrom_LineEdit.text()), float(self.macro_UI.learningRateTo_LineEdit.text()), float(self.macro_UI.learningRateStep_LineEdit.text())));
        modified_Variable_List.append(float(self.macro_UI.learningRateTo_LineEdit.text()));

        self.current_Modify_Factor_List.append(("Learning_Rate", None, modified_Variable_List));

        self.Macro_UI_Modify_Factor_Changed();

    def Macro_UI_initialWeightSDAdd_Button_Clicked(self):
        if self.macro_UI.initialWeightSDFrom_LineEdit.text() == "":
            self.macro_UI.initialWeightSDFrom_LineEdit.setFocus();
            return;
        elif self.macro_UI.initialWeightSDTo_LineEdit.text() == "":
            self.macro_UI.initialWeightSDTo_LineEdit.setFocus();
            return;
        elif self.macro_UI.initialWeightSDStep_LineEdit.text() == "":
            self.macro_UI.initialWeightSDStep_LineEdit.setFocus();
            return;

        for modify_Type, factor, variable_List in self.current_Modify_Factor_List:
            if modify_Type == "Initial_Weight_SD":
                QtWidgets.QMessageBox.warning(None, 'Warning!', "Initial weight SD is already in the list as a modify factor.");
                return;

        modified_Variable_List = [];
        modified_Variable_List.extend(np.arange(float(self.macro_UI.initialWeightSDFrom_LineEdit.text()), float(self.macro_UI.initialWeightSDTo_LineEdit.text()), float(self.macro_UI.initialWeightSDStep_LineEdit.text())));
        modified_Variable_List.append(float(self.macro_UI.initialWeightSDTo_LineEdit.text()));

        self.current_Modify_Factor_List.append(("Initial_Weight_SD", None, modified_Variable_List));

        self.Macro_UI_Modify_Factor_Changed();
        
    def Macro_UI_layerDamageSDAdd_Button_Clicked(self):
        if self.macro_UI.layerDamageSDFrom_LineEdit.text() == "":
            self.macro_UI.layerDamageSDFrom_LineEdit.setFocus();
            return;
        elif self.macro_UI.layerDamageSDTo_LineEdit.text() == "":
            self.macro_UI.layerDamageSDTo_LineEdit.setFocus();
            return;
        elif self.macro_UI.layerDamageSDStep_LineEdit.text() == "":
            self.macro_UI.layerDamageSDStep_LineEdit.setFocus();
            return;

        selected_Process = self.macro_UI.layerDamageSDProcess_ComboBox.currentText();
        selected_Layer = self.macro_UI.layerDamageSDLayer_ComboBox.currentText()

        for modify_Type, factor, variable_List in self.current_Modify_Factor_List:
            if modify_Type == "Layer_Damage_SD" and factor[0] == selected_Process and factor[1] == selected_Layer:
                QtWidgets.QMessageBox.warning(None, 'Warning!', "This layer of this process is already in the list as a modify factor.");
                return;

        modified_Variable_List = [];
        if self.macro_UI.layerDamageSDOn_CheckBox.isChecked():
            modified_Variable_List.append("On");
        if self.macro_UI.layerDamageSDOff_CheckBox.isChecked():
            modified_Variable_List.append("Off");
        modified_Variable_List.extend(np.arange(float(self.macro_UI.layerDamageSDFrom_LineEdit.text()), float(self.macro_UI.layerDamageSDTo_LineEdit.text()), float(self.macro_UI.layerDamageSDStep_LineEdit.text())));
        modified_Variable_List.append(float(self.macro_UI.layerDamageSDTo_LineEdit.text()));

        self.current_Modify_Factor_List.append(("Layer_Damage_SD", (selected_Process, selected_Layer), modified_Variable_List));

        self.Macro_UI_Modify_Factor_Changed();
        
    def Macro_UI_connectionDamageSDAdd_Button_Clicked(self):
        if self.macro_UI.connectionDamageSDFrom_LineEdit.text() == "":
            self.macro_UI.connectionDamageSDFrom_LineEdit.setFocus();
            return;
        elif self.macro_UI.connectionDamageSDTo_LineEdit.text() == "":
            self.macro_UI.connectionDamageSDTo_LineEdit.setFocus();
            return;
        elif self.macro_UI.connectionDamageSDStep_LineEdit.text() == "":
            self.macro_UI.connectionDamageSDStep_LineEdit.setFocus();
            return;

        selected_Process = self.macro_UI.connectionDamageSDProcess_ComboBox.currentText();
        selected_Connection = self.macro_UI.connectionDamageSDConnection_ComboBox.currentText();

        for modify_Type, factor, variable_List in self.current_Modify_Factor_List:
            if modify_Type == "Connection_Damage_SD" and factor[0] == selected_Process and factor[1] == selected_Connection:
                QtWidgets.QMessageBox.warning(None, 'Warning!', "This connection of this process is already in the list as a modify factor.");
                return;

        modified_Variable_List = [];
        if self.macro_UI.connectionDamageSDOn_CheckBox.isChecked():
            modified_Variable_List.append("On");
        if self.macro_UI.connectionDamageSDOff_CheckBox.isChecked():
            modified_Variable_List.append("Off");
        modified_Variable_List.extend(np.arange(float(self.macro_UI.connectionDamageSDFrom_LineEdit.text()), float(self.macro_UI.connectionDamageSDTo_LineEdit.text()), float(self.macro_UI.connectionDamageSDStep_LineEdit.text())));
        modified_Variable_List.append(float(self.macro_UI.connectionDamageSDTo_LineEdit.text()));

        self.current_Modify_Factor_List.append(("Connection_Damage_SD", (selected_Process, selected_Connection), modified_Variable_List));

        self.Macro_UI_Modify_Factor_Changed();
        
    def Macro_UI_learning_Button_Clicked(self):
        self.macro_Dialog.hide();
        self.learning_Dialog.show();
        self.macro_Index = 0;
        self.simulator = self.modified_Simulator_List[0];
        self.simulator.Weight_and_Bias_Setup();
        self.simulator.Process_To_Tensor();
        self.Learning_UI_start_Button_Clicked();
        self.learning_UI.macro_LineEdit.setText("Macro 0");        
        self.learning_UI.macro_Label.show();
        self.learning_UI.macro_LineEdit.show();
        
        self.tf_graph = tf.get_default_graph();
        thread.start_new_thread(self.Macro_UI_Simulator_Finished_Check, ());
    
    def Macro_UI_exit_Button_Clicked(self):
        self.macro_Dialog.hide();
        self.main_Dialog.exec_();
        
    def Macro_UI_base_Model_Changed(self):
        self.modified_Simulator_List = [];
        self.current_Modify_Factor_List = [];

        self.macro_UI.layer_ListWidget.clear();
        self.macro_UI.connection_ListWidget.clear();
        self.macro_UI.patternPack_ListWidget.clear();
        self.macro_UI.process_ListWidget.clear();
        self.macro_UI.learningSetup_ListWidget.clear();
        self.macro_UI.layerSizeLayer_ComboBox.clear();
        self.macro_UI.layerDamageSDProcess_ComboBox.clear();
        self.macro_UI.layerDamageSDLayer_ComboBox.clear();
        self.macro_UI.connectionDamageSDProcess_ComboBox.clear();
        self.macro_UI.connectionDamageSDConnection_ComboBox.clear();

        config_Dict_Display_List = [];
        config_Dict_Display_List.append("Momentum: " + str(self.macro_Base_Dict["Config_Dict"]["Momentum"]));
        config_Dict_Display_List.append("Learning Rate: " + str(self.macro_Base_Dict["Config_Dict"]["Learning_Rate"]));
        config_Dict_Display_List.append("Decay Rate: " + str(self.macro_Base_Dict["Config_Dict"]["Decay_Rate"]));
        config_Dict_Display_List.append("Initial Weight SD: " + str(self.macro_Base_Dict["Config_Dict"]["Initial_Weight_SD"]));
        config_Dict_Display_List.append("Device Mode: " + str(self.macro_Base_Dict["Config_Dict"]["Device_Mode"]).upper());
        self.macro_UI.configVariables_TextEdit.setText("\n".join(config_Dict_Display_List));

        for key in self.macro_Base_Dict["Layer_Dict"].keys():
            layer_Information = self.macro_Base_Dict["Layer_Dict"][key];
            self.macro_UI.layer_ListWidget.addItem(key + " (" + str(layer_Information["Unit"]) + ")");            

        for key in self.macro_Base_Dict["Connection_Dict"].keys():
            connection_Information = self.macro_Base_Dict["Connection_Dict"][key];
            self.macro_UI.connection_ListWidget.addItem(key + " (" + str(connection_Information["From_Layer_Name"]) + "→" + str(connection_Information["To_Layer_Name"]) + ")");

        for key in self.macro_Base_Dict["Pattern_Pack_Dict"].keys():
            self.macro_UI.patternPack_ListWidget.addItem(key);

        for key in self.macro_Base_Dict["Process_Dict"].keys():
            self.macro_UI.process_ListWidget.addItem(key);

        for learning_Setup in self.macro_Base_Dict["Learning_Setup_List"]:
            self.macro_UI.learningSetup_ListWidget.addItem(learning_Setup["Name"]);

        for key in self.macro_Base_Dict["Layer_Dict"].keys():
            self.macro_UI.layerSizeLayer_ComboBox.addItem(key);

        for key in self.macro_Base_Dict["Process_Dict"].keys():
            self.macro_UI.layerDamageSDProcess_ComboBox.addItem(key);

        for key in self.macro_Base_Dict["Layer_Dict"].keys():
            self.macro_UI.layerDamageSDLayer_ComboBox.addItem(key);

        for key in self.macro_Base_Dict["Process_Dict"].keys():
            self.macro_UI.connectionDamageSDProcess_ComboBox.addItem(key);

        for key in self.macro_Base_Dict["Connection_Dict"].keys():
            self.macro_UI.connectionDamageSDConnection_ComboBox.addItem(key);
        
        self.macro_UI.macroInformation_GroupBox.setEnabled(True);
        self.macro_UI.modifyFactor_GroupBox.setEnabled(True);

        self.Macro_UI_Modify_Factor_Changed();
    
    def Macro_UI_Modified_Simulator_Changed(self):
        self.macro_UI.macro_ListWidget.clear();

        for index in range(len(self.modified_Simulator_List)):
            self.macro_UI.macro_ListWidget.addItem("Macro_" + str(index));

    def Macro_UI_Modify_Factor_Changed(self):
        self.macro_UI.modifyFactor_ListWidget.clear();
        self.macro_UI.macroSize_LineEdit.setText("");        
        self.macro_UI.layerSizeLayer_ComboBox.setCurrentIndex(0);
        self.macro_UI.layerDamageSDProcess_ComboBox.setCurrentIndex(0);
        self.macro_UI.layerDamageSDLayer_ComboBox.setCurrentIndex(0);
        self.macro_UI.connectionDamageSDConnection_ComboBox.setCurrentIndex(0);
        self.macro_UI.connectionDamageSDProcess_ComboBox.setCurrentIndex(0);

        self.macro_UI.layerSizeFrom_LineEdit.setText("");
        self.macro_UI.layerSizeTo_LineEdit.setText("");
        self.macro_UI.layerSizeStep_LineEdit.setText("");
        self.macro_UI.regularMultiLayerSizeLayerPrefix_LineEdit.setText("");
        self.macro_UI.regularMultiLayerSizeMaxSuffix_LineEdit.setText("");
        self.macro_UI.regularMultiLayerSizeFrom_LineEdit.setText("");
        self.macro_UI.regularMultiLayerSizeTo_LineEdit.setText("");
        self.macro_UI.regularMultiLayerSizeStep_LineEdit.setText("");
        self.macro_UI.irregularMultiLayerSizeLayer_LineEdit.setText("");
        self.macro_UI.irregularMultiLayerSizeFrom_LineEdit.setText("");
        self.macro_UI.irregularMultiLayerSizeTo_LineEdit.setText("");
        self.macro_UI.irregularMultiLayerSizeStep_LineEdit.setText("");
        self.macro_UI.learningRateFrom_LineEdit.setText("");
        self.macro_UI.learningRateTo_LineEdit.setText("");
        self.macro_UI.learningRateStep_LineEdit.setText("");
        self.macro_UI.initialWeightSDFrom_LineEdit.setText("");
        self.macro_UI.initialWeightSDTo_LineEdit.setText("");
        self.macro_UI.initialWeightSDStep_LineEdit.setText("");
        self.macro_UI.layerDamageSDFrom_LineEdit.setText("");
        self.macro_UI.layerDamageSDTo_LineEdit.setText("");
        self.macro_UI.layerDamageSDStep_LineEdit.setText("");
        self.macro_UI.layerDamageSDOn_CheckBox.setChecked(True);
        self.macro_UI.layerDamageSDOff_CheckBox.setChecked(False);
        self.macro_UI.connectionDamageSDFrom_LineEdit.setText("");
        self.macro_UI.connectionDamageSDTo_LineEdit.setText("");
        self.macro_UI.connectionDamageSDStep_LineEdit.setText("");
        self.macro_UI.connectionDamageSDOn_CheckBox.setChecked(True);
        self.macro_UI.connectionDamageSDOff_CheckBox.setChecked(False);

        size_Text_List = [];
        size_Value = 1;
        for modify_Factor in self.current_Modify_Factor_List:
            if modify_Factor[0] == "Layer_Size":                
                self.macro_UI.modifyFactor_ListWidget.addItem("Layer Size: " + modify_Factor[1] + "(" + ", ".join([str(x) for x in modify_Factor[2]]) + ")");
            elif modify_Factor[0] == "Multi_Layer_Size":                
                self.macro_UI.modifyFactor_ListWidget.addItem("Layer Size: " + str(modify_Factor[1]) + "(" + ", ".join([str(x) for x in modify_Factor[2]]) + ")");
            elif modify_Factor[0] == "Learning_Rate":
                self.macro_UI.modifyFactor_ListWidget.addItem("Learning Rate: (" + ", ".join([str(x) for x in modify_Factor[2]]) + ")");
            elif modify_Factor[0] == "Initial_Weight_SD":
                self.macro_UI.modifyFactor_ListWidget.addItem("Initial Weight SD: (" + ", ".join([str(x) for x in modify_Factor[2]]) + ")");
            elif modify_Factor[0] == "Layer_Damage_SD":
                self.macro_UI.modifyFactor_ListWidget.addItem("Layer Damage SD: (P: " + modify_Factor[1][0] + ", L: " + modify_Factor[1][1] + ") (" + ", ".join([str(x) for x in modify_Factor[2]]) + ")");
            elif modify_Factor[0] == "Connection_Damage_SD":
                self.macro_UI.modifyFactor_ListWidget.addItem("Connection Damage SD: (P: " + modify_Factor[1][0] + ", C: " + modify_Factor[1][1] + ") (" + ", ".join([str(x) for x in modify_Factor[2]]) + ")");

            size_Text_List.append(str(len(modify_Factor[2])));
            size_Value *= len(modify_Factor[2]);
        
        if len(size_Text_List) == 0:
            self.macro_UI.macroSize_LineEdit.setText("0");
        else:
            self.macro_UI.macroSize_LineEdit.setText(" × ".join(size_Text_List) + " = " + str(size_Value));

    def Macro_UI_Modifying_Factor_Apply(self, base_Dict_List, modify_Factor):
        print(modify_Factor);
        modified_Dict_List = [];
        for modified_Variable in modify_Factor[2]:
            for base_Dict in base_Dict_List:
                new_Data = deepcopy(base_Dict);
                if modify_Factor[0] == "Layer_Size":
                    new_Data["Layer_Dict"][modify_Factor[1]]["Unit"] = modified_Variable;
                    for connection_Key in new_Data["Connection_Dict"].keys():
                        if new_Data["Connection_Dict"][connection_Key]["From_Layer_Name"] == modify_Factor[1]:
                            new_Data["Connection_Dict"][connection_Key]["From_Layer_Unit"] = modified_Variable;
                        if new_Data["Connection_Dict"][connection_Key]["To_Layer_Name"] == modify_Factor[1]:
                            new_Data["Connection_Dict"][connection_Key]["To_Layer_Unit"] = modified_Variable;
                elif modify_Factor[0] == "Multi_Layer_Size":
                    for layer_Name in modify_Factor[1]:
                        new_Data["Layer_Dict"][layer_Name]["Unit"] = modified_Variable;
                        for connection_Key in new_Data["Connection_Dict"].keys():
                            if new_Data["Connection_Dict"][connection_Key]["From_Layer_Name"] == layer_Name:
                                new_Data["Connection_Dict"][connection_Key]["From_Layer_Unit"] = modified_Variable;
                            if new_Data["Connection_Dict"][connection_Key]["To_Layer_Name"] == layer_Name:
                                new_Data["Connection_Dict"][connection_Key]["To_Layer_Unit"] = modified_Variable;                    
                elif modify_Factor[0] == "Learning_Rate":
                    new_Data["Config_Dict"]["Learning_Rate"] = modified_Variable;
                elif modify_Factor[0] == "Initial_Weight_SD":
                    new_Data["Config_Dict"]["Initial_Weight_SD"] = modified_Variable;
                elif modify_Factor[0] == "Layer_Damage_SD":
                    if modified_Variable == "On":
                        new_Data["Process_Dict"][modify_Factor[1][0]]["Layer_Control_Dict"][modify_Factor[1][1]] = (Damage_Type.On, None)
                    elif modified_Variable == "Off":
                        new_Data["Process_Dict"][modify_Factor[1][0]]["Layer_Control_Dict"][modify_Factor[1][1]] = (Damage_Type.Off, None)
                    else:                    
                        new_Data["Process_Dict"][modify_Factor[1][0]]["Layer_Control_Dict"][modify_Factor[1][1]] = (Damage_Type.Damaged, modified_Variable)
                elif modify_Factor[0] == "Connection_Damage_SD":
                    if modified_Variable == "On":
                        new_Data["Process_Dict"][modify_Factor[1][0]]["Connection_Control_Dict"][modify_Factor[1][1]] = (Damage_Type.On, None)
                    elif modified_Variable == "Off":
                        new_Data["Process_Dict"][modify_Factor[1][0]]["Connection_Control_Dict"][modify_Factor[1][1]] = (Damage_Type.Off, None)
                    else:                    
                        new_Data["Process_Dict"][modify_Factor[1][0]]["Connection_Control_Dict"][modify_Factor[1][1]] = (Damage_Type.Damaged, modified_Variable)                
                modified_Dict_List.append(new_Data);
        
        return modified_Dict_List;
        
    def Macro_UI_Simulator_Finished_Check(self):
        while True:
            time.sleep(1.0);
            if self.simulator.current_Learning_Setup_Index < len(self.simulator.learning_Setup_List):                    
                continue;

            self.macro_Index += 1;
            if self.macro_Index < len(self.modified_Simulator_List):                
                tf.reset_default_graph();
                with self.tf_graph.as_default():                    
                    self.simulator = self.modified_Simulator_List[self.macro_Index];
                    self.simulator.Weight_and_Bias_Setup();
                    self.simulator.Process_To_Tensor();
                self.Learning_UI_start_Button_Clicked();
                self.learning_UI.macro_LineEdit.setText("Macro " + str(self.macro_Index));
            else:
                break;

    # End Macro Functions

    # Start About Functions
    def About_Setup_UI_exit_Button_Clicked(self):
        self.about_Dialog.hide();
        self.main_Dialog.exec_();

    # End About Functions

if __name__ == "__main__":
    hNet_GUI = HNet_GUI();
