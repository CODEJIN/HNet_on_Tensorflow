###############################################################################
# HNet with Tensorflow Core
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2017 Heejo You
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
###############################################################################

from HNet_Enum import *;
import numpy as np;
import tensorflow as tf;
import time, os, io;
from random import shuffle; 
import _pickle as pickle;
from tensorflow.python.client import device_lib;

class HNet:
    def __init__(self):
        self.tf_Session = tf.Session();

        self.config_Variables_Dict = {};
        self.layer_Information_Dict = {};
        self.connection_Information_Dict = {};
        self.pattern_Pack_Dict = {};
        self.process_Dict = {};
        self.learning_Setup_List = [];
        
        self.Structure_Config_Variable_Setup(0.9, 0.0, 0.1, 0.01, 'cpu');

        self.extract_Result_Dict = {};

        self.current_Total_Epoch = 0;
        self.current_Learning_Setup_Index = 0;
        self.current_LearningSetup_Epoch = 0;
        self.pause_Status = True;

    def Structure_Config_Variable_Setup(self, momentum, decay_Rate, initial_Weight_SD, learning_Rate, device_Mode):
        self.config_Variables_Dict["Momentum"] = momentum;
        self.config_Variables_Dict["Decay_Rate"] = decay_Rate;
        self.config_Variables_Dict["Initial_Weight_SD"] = initial_Weight_SD;
        self.config_Variables_Dict["Learning_Rate"] = learning_Rate;
        self.config_Variables_Dict["Device_Mode"] = device_Mode #'cpu' or 'gpu'

    def Structure_Layer_Assign(self, name, unit):
        new_Layer_Inforamtion_Dict = {};
        new_Layer_Inforamtion_Dict["Unit"] = unit;

        self.layer_Information_Dict[name] = new_Layer_Inforamtion_Dict;

    def Structure_Connection_Assign(self, name, from_Layer_Name, to_Layer_Name):
        new_Connection_Information_Dict = {};
        new_Connection_Information_Dict["From_Layer_Name"] = from_Layer_Name;
        new_Connection_Information_Dict["To_Layer_Name"] = to_Layer_Name;
        new_Connection_Information_Dict["From_Layer_Unit"] = self.layer_Information_Dict[from_Layer_Name]["Unit"];
        new_Connection_Information_Dict["To_Layer_Unit"] = self.layer_Information_Dict[to_Layer_Name]["Unit"];
        
        self.connection_Information_Dict[name] = new_Connection_Information_Dict;

    def Structure_Layer_Delete(self, name):
        del self.layer_Information_Dict[name];

        delete_Connection_Key_List = [];
        for connection_Key in self.connection_Information_Dict:
            if self.connection_Information_Dict[connection_Key]["From_Layer_Name"] == name or self.connection_Information_Dict[connection_Key]["To_Layer_Name"] == name:
                delete_Connection_Key_List.append(connection_Key);
        for connection_Key in delete_Connection_Key_List:
            self.Structure_Connection_Delete(connection_Key);

    def Structure_Connection_Delete(self, name):
        del self.connection_Information_Dict[name];

    def Structure_Save(self, file_Path):
        save_Dict = {};

        save_Dict["Config_Variables_Dict"] = self.config_Variables_Dict;
        save_Dict["Layer_Dict"] = self.layer_Information_Dict;         
        save_Dict["Connection_Dict"] = self.connection_Information_Dict;

        if file_Path[-15:] != ".HNet_Structure":
            file_Path += ".HNet_Structure";

        with open(file_Path, "wb") as f:
            pickle.dump(save_Dict, f);

    def Structure_Load(self, file_Path):
        load_Dict = {};
        with open(file_Path, "rb") as f:
            load_Dict = pickle.load(f);

        self.config_Variables_Dict = load_Dict["Config_Variables_Dict"];
        self.layer_Information_Dict = load_Dict["Layer_Dict"];
        self.connection_Information_Dict = load_Dict["Connection_Dict"];

        if len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU']) < 1 and self.config_Variables_Dict["Device_Mode"] == 'gpu':            
            if __name__ == "__main__":
                print("This enviroment cannot support 'GPU'. Device mode was changed to 'CPU''");
            self.config_Variables_Dict["Device_Mode"] = 'cpu';            

    def Pattern_Pack_Load(self, name, file_Path):
        new_Pattern_Pack_Dict = {};

        with open(file_Path, "r", encoding="utf8") as f:
            readLines = f.readlines();

        column_Name_List = readLines[0].replace("\n", "").strip().split("\t");
        column_Name_Dict = {};
        for column_Index in range(len(column_Name_List)):
            column_Name_Dict[column_Index] = column_Name_List[column_Index];
            new_Pattern_Pack_Dict[column_Name_List[column_Index]] = [];

        new_Pattern_Pack_Dict["Count"] = len(readLines[1:]);

        for readLine in readLines[1:]:
            pattern_Data = readLine.replace("\n", "").strip().split("\t");
            for column_Index in range(len(pattern_Data)):                
                column_Name = column_Name_Dict[column_Index];
                pattern = pattern_Data[column_Index];

                if column_Name == "Name":
                    new_Pattern_Pack_Dict[column_Name].append(pattern);
                elif column_Name == "Probability" or column_Name == "Cycle":
                    new_Pattern_Pack_Dict[column_Name].append(float(pattern));
                else:                    
                    new_Pattern_Pack_Dict[column_Name].append(np.array([float(x) for x in pattern.strip().split(" ")]));

        for pattern_Key in new_Pattern_Pack_Dict.keys():
            if pattern_Key == "Name" or pattern_Key == "Count":
                continue;
            elif pattern_Key == "Probability" or pattern_Key == "Cycle":            
                new_Pattern_Pack_Dict[pattern_Key] = np.array(new_Pattern_Pack_Dict[pattern_Key]).reshape(len(new_Pattern_Pack_Dict[pattern_Key]), 1);
            else:
                new_Pattern_Pack_Dict[pattern_Key] = np.array(new_Pattern_Pack_Dict[pattern_Key]);

        self.pattern_Pack_Dict[name] = new_Pattern_Pack_Dict;

    def Pattern_Pack_Delete(self, name):
        del self.pattern_Pack_Dict[name];
        
    def Process_Assign(self, name, order_List, layer_Control_Dict, connection_Control_Dict):
        new_process_Dict = {};
        new_process_Dict["Order_List"] = order_List; #(order_Code, layer_Name_List, connection_Name_List)
        new_process_Dict["Layer_Control_Dict"] = layer_Control_Dict; #(type, SD)
        new_process_Dict["Connection_Control_Dict"] = connection_Control_Dict; #(type, SD)
        self.process_Dict[name] = new_process_Dict;

    def Process_Delete(self, name):
        del self.process_Dict[name];

    def Process_Save(self, file_Path):
        save_Dict = {};

        save_Dict["Process_Dict"] = self.process_Dict;

        if file_Path[-13:] != ".HNet_Process":
            file_Path += ".HNet_Process";
        with open(file_Path, "wb") as f:
            pickle.dump(save_Dict, f);

    def Process_Load(self, file_Path):
        load_Dict = {};
        with open(file_Path, "rb") as f:
            load_Dict = pickle.load(f);

        for process_Name in load_Dict["Process_Dict"].keys():
            process = load_Dict["Process_Dict"][process_Name];
            if set(process["Layer_Control_Dict"].keys()) != set(self.layer_Information_Dict.keys()) or set(process["Connection_Control_Dict"].keys()) != set(self.connection_Information_Dict.keys()):
                if __name__ == "__main__":
                    print("Current structure is not compatiable with this process file.");
                return False;

        self.process_Dict = load_Dict["Process_Dict"];
        return True;

    def Learning_Setup_Assign(self, name, training_Matching_List, test_Matching_List, training_Epoch, test_Timing, mini_Batch_Size, shuffle_Mode):
        new_Learning_Setup = {};

        new_Learning_Setup["Name"] = name
        new_Learning_Setup["Training_Matching_List"] = training_Matching_List; #The List of Pattern_Matching
        new_Learning_Setup["Test_Matching_List"] = test_Matching_List; #The List of Pattern_Matching
        new_Learning_Setup["Training_Epoch"] = training_Epoch;
        new_Learning_Setup["Test_Timing"] = test_Timing;
        new_Learning_Setup["Mini_Batch_Size"] = mini_Batch_Size;
        new_Learning_Setup["Shuffle_Mode"] = shuffle_Mode;

        for index in range(len(self.learning_Setup_List)):
            if self.learning_Setup_List[index]["Name"] == name:
                self.learning_Setup_List[index] = new_Learning_Setup;
                return;

        self.learning_Setup_List.append(new_Learning_Setup);

    def Learning_Setup_Delete(self, index):
        del self.learning_Setup_List[index];

    def Learning_Setup_Save(self, file_Path):
        save_Dict = {};

        save_Dict["Learning_Setup_List"] = self.learning_Setup_List;

        if file_Path[-20:] != ".HNet_Learning_Setup":
            file_Path += ".HNet_Learning_Setup";

        with open(file_Path, "wb") as f:
            pickle.dump(save_Dict, f);

    def Learning_Setup_Load(self, file_Path):
        load_Dict = {};
        with open(file_Path, "rb") as f:
            load_Dict = pickle.load(f);

        consistency = True;
        for learning_Setup in load_Dict["Learning_Setup_List"]:            
            for matching_Information in learning_Setup["Training_Matching_List"]:                
                process_Name = matching_Information["Process_Name"];
                pattern_Pack_Name = matching_Information["Pattern_Pack_Name"];
                assign_Dict = matching_Information["Assign"];
                if not process_Name in self.process_Dict.keys():
                    consistency = False;
                    break;
                elif not pattern_Pack_Name in self.pattern_Pack_Dict.keys():
                    consistency = False;
                    break;
                used_Layer_Set = set();
                for order_Index in assign_Dict.keys():
                    if not assign_Dict[order_Index] in self.pattern_Pack_Dict[pattern_Pack_Name]:
                        consistency = False;
                        break;                    
                    order_Code, layer_Name_List, connection_Name_List, order_Variable_List = self.process_Dict[process_Name]["Order_List"][order_Index];
                    if not order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax]:
                        consistency = False;
                        break;
                    else:
                        used_Layer_Set.add(layer_Name_List[0]);
                if used_Layer_Set & set(self.layer_Information_Dict.keys()) != used_Layer_Set:
                    consistency = False;
                    break;
            if not consistency:
                break;
            for matching_Information in learning_Setup["Test_Matching_List"]:
                process_Name = matching_Information["Process_Name"];
                pattern_Pack_Name = matching_Information["Pattern_Pack_Name"];
                assign_Dict = matching_Information["Assign"];
                extract_Data_List = matching_Information["Extract_Data"];
                if not process_Name in self.process_Dict.keys():
                    consistency = False;
                    break;
                elif not pattern_Pack_Name in self.pattern_Pack_Dict.keys():
                    consistency = False;
                    break;
                used_Layer_Set = set();
                for order_Index in assign_Dict.keys():
                    if not assign_Dict[order_Index] in self.pattern_Pack_Dict[pattern_Pack_Name]:
                        consistency = False;
                        break;                    
                    order_Code, layer_Name_List, connection_Name_List, order_Variable_List = self.process_Dict[process_Name]["Order_List"][order_Index];
                    if not order_Code in [Order_Code.Input_Layer_Acitvation_Insert, Order_Code.Output_Layer_Error_Calculation_Sigmoid, Order_Code.Output_Layer_Error_Calculation_Softmax]:
                        consistency = False;
                        break;
                    else:
                        used_Layer_Set.add(layer_Name_List[0]);
                if used_Layer_Set & set(self.layer_Information_Dict.keys()) != used_Layer_Set:
                    consistency = False;
                    break;
                used_Layer_Set = set();
                for pattern, order_Index, extract_Data_Type in extract_Data_List:
                    if not pattern is None and not pattern in self.pattern_Pack_Dict[pattern_Pack_Name]:
                        consistency = False;
                        break;
                    order_Code, layer_Name_List, connection_Name_List, order_Variable_List = self.process_Dict[process_Name]["Order_List"][order_Index];
                    if not order_Code == Order_Code.Activation_Extract:
                        consistency = False;
                        break;
                    else:
                        used_Layer_Set.add(layer_Name_List[0]);
                if used_Layer_Set & set(self.layer_Information_Dict.keys()) != used_Layer_Set:
                    consistency = False;
                    break;
            if not consistency:
                break;

        if not consistency:
            if __name__ == "__main__":
                print("Current structure or process is not compatiable with this learning setup file.");
            return False;

        self.learning_Setup_List = load_Dict["Learning_Setup_List"];
        return True;

    def WeightAndBias_Save(self, file_Path):        
        save_Dict = {};
        save_Dict["Weight_Dict"] = {};
        save_Dict["Bias_Dict"] = {};

        for key in self.weightMatrix_Dict.keys():
            save_Dict["Weight_Dict"][key], = self.tf_Session.run([self.weightMatrix_Dict[key]]);
        
        for key in self.biasMatrix_Dict.keys():
            save_Dict["Bias_Dict"][key], = self.tf_Session.run([self.biasMatrix_Dict[key]]);

        if file_Path[-11:] != ".HNet_Model":
            file_Path += ".HNet_Model";
        with open(file_Path, "wb") as f:
            pickle.dump(save_Dict, f);

    def WeightAndBias_Load(self, file_Path):
        load_Dict = {};
        with open(file_Path, "rb") as f:
            load_Dict = pickle.load(f);

        #Consistency Check
        if len(load_Dict["Weight_Dict"]) != len(self.connection_Information_Dict):
            return False;

        for connection_Key in self.connection_Information_Dict.keys():
            connection_Information = self.connection_Information_Dict[connection_Key];
            if not connection_Key in load_Dict["Weight_Dict"].keys() or not load_Dict["Weight_Dict"][connection_Key].shape == (connection_Information["From_Layer_Unit"], connection_Information["To_Layer_Unit"]):
                return False;

        if len(load_Dict["Bias_Dict"]) != len(self.layer_Information_Dict):
            return False;

        for layer_Key in self.layer_Information_Dict.keys():
            layer_Information = self.layer_Information_Dict[layer_Key];            
            if not layer_Key in load_Dict["Bias_Dict"].keys() or not load_Dict["Bias_Dict"][layer_Key].shape == (1, layer_Information["Unit"]):
                return False;

        with tf.device('/' + self.config_Variables_Dict["Device_Mode"]):
            for connection_Key in load_Dict["Weight_Dict"].keys():
                saved_Weight = load_Dict["Weight_Dict"][connection_Key];
                placeHolder = tf.placeholder(tf.float32);
                self.tf_Session.run([tf.assign(self.weightMatrix_Dict[connection_Key], placeHolder)], feed_dict = {placeHolder: saved_Weight.astype("float32")});

            for layer_Key in load_Dict["Bias_Dict"].keys():
                saved_Bias = load_Dict["Bias_Dict"][layer_Key];
                placeHolder = tf.placeholder(tf.float32);
                self.tf_Session.run([tf.assign(self.biasMatrix_Dict[layer_Key], placeHolder)], feed_dict = {placeHolder: saved_Bias.astype("float32")});

        return True;
    
    def Weight_and_Bias_Setup(self):
        self.weightMatrix_Dict = {};
        self.biasMatrix_Dict = {};
        
        with tf.device('/' + self.config_Variables_Dict["Device_Mode"]):
            for connection_Key in self.connection_Information_Dict.keys():
                connection_Information = self.connection_Information_Dict[connection_Key];
                self.weightMatrix_Dict[connection_Key] = tf.Variable(tf.random_normal((connection_Information["From_Layer_Unit"], connection_Information["To_Layer_Unit"]), 0, self.config_Variables_Dict["Initial_Weight_SD"]));
                self.tf_Session.run(tf.variables_initializer([self.weightMatrix_Dict[connection_Key]]));

            for layer_Key in self.layer_Information_Dict.keys():
                layer_Information = self.layer_Information_Dict[layer_Key];            
                self.biasMatrix_Dict[layer_Key] = tf.Variable(tf.random_normal((1, layer_Information["Unit"]), 0, self.config_Variables_Dict["Initial_Weight_SD"]));
                self.tf_Session.run(tf.variables_initializer([self.biasMatrix_Dict[layer_Key]]));
    
    def Process_To_Tensor(self):
        with tf.device('/' + self.config_Variables_Dict["Device_Mode"]):        
            for process_Key in self.process_Dict.keys():
                process = self.process_Dict[process_Key];
                
                process["PlaceHolder_Dict"] = {};
                process["Tensor_List"] = [];
                process["Extract_Activation_Tensor_Index_Dict"] = {};
                process["Extract_Activation_Tensor_Cycle_Dict"] = {};
                
                process["PlaceHolder_Dict"]["Probability_Filter"] = tf.placeholder(tf.float32);
                process["PlaceHolder_Dict"]["Cycle_Filter"] = tf.placeholder(tf.float32);
                current_Cycle = 0;

                layer_Activation_Stroage_Dict = {};
                layer_Activation_Dict = {};
                layer_Error_Stroage_Dict = {};
                layer_Error_Dict = {};

                for order_Index in range(len(process["Order_List"])):
                    order_Code, layer_Name_List, connection_Name_List, order_Variable_List = process["Order_List"][order_Index];
                    if order_Code == Order_Code.Input_Layer_Acitvation_Insert:
                        process["PlaceHolder_Dict"][order_Index] = tf.placeholder(tf.float32);
                        damage_Type, SD = process["Layer_Control_Dict"][layer_Name_List[0]];
                        if damage_Type == Damage_Type.On:
                            layer_Activation_Dict[layer_Name_List[0]] = process["PlaceHolder_Dict"][order_Index] * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);
                        elif damage_Type == Damage_Type.Off:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.zeros(tf.shape(process["PlaceHolder_Dict"][order_Index]));
                        elif damage_Type == Damage_Type.Damaged:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(process["PlaceHolder_Dict"][order_Index] + tf.random_normal(tf.shape(process["PlaceHolder_Dict"][order_Index]), 0, SD), 0, 1) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);
                    
                    elif order_Code == Order_Code.Activation_Calculation_Sigmoid:
                        damage_Type, SD = process["Layer_Control_Dict"][layer_Name_List[0]];
                        if damage_Type == Damage_Type.On:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.sigmoid((layer_Activation_Stroage_Dict[layer_Name_List[0]] + self.biasMatrix_Dict[layer_Name_List[0]]) * self.config_Variables_Dict["Momentum"]) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);
                        elif damage_Type == Damage_Type.Off:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.zeros(tf.shape(layer_Activation_Stroage_Dict[layer_Name_List[0]]));
                        elif damage_Type == Damage_Type.Damaged:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(tf.sigmoid((layer_Activation_Stroage_Dict[layer_Name_List[0]] + self.biasMatrix_Dict[layer_Name_List[0]]) * self.config_Variables_Dict["Momentum"]) + tf.random_normal(tf.shape(layer_Activation_Stroage_Dict[layer_Name_List[0]]), 0, SD), 0, 1) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Activation_Calculation_Softmax:
                        damage_Type, SD = process["Layer_Control_Dict"][layer_Name_List[0]];
                        if damage_Type == Damage_Type.On:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.nn.softmax(layer_Activation_Stroage_Dict[layer_Name_List[0]] + self.biasMatrix_Dict[layer_Name_List[0]]) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);
                        elif damage_Type == Damage_Type.Off:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.zeros(tf.shape(layer_Activation_Stroage_Dict[layer_Name_List[0]]));
                        elif damage_Type == Damage_Type.Damaged:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(tf.nn.softmax(layer_Activation_Stroage_Dict[layer_Name_List[0]] + self.biasMatrix_Dict[layer_Name_List[0]]) + tf.random_normal(tf.shape(layer_Activation_Stroage_Dict[layer_Name_List[0]]), 0, SD), 0, 1) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Activation_Calculation_ReLU:
                        damage_Type, SD = process["Layer_Control_Dict"][layer_Name_List[0]];
                        if damage_Type == Damage_Type.On:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.nn.relu(layer_Activation_Stroage_Dict[layer_Name_List[0]] + self.biasMatrix_Dict[layer_Name_List[0]]) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);
                        elif damage_Type == Damage_Type.Off:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.zeros(tf.shape(layer_Activation_Stroage_Dict[layer_Name_List[0]]));
                        elif damage_Type == Damage_Type.Damaged:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(tf.nn.relu(layer_Activation_Stroage_Dict[layer_Name_List[0]] + self.biasMatrix_Dict[layer_Name_List[0]]) + tf.random_normal(tf.shape(layer_Activation_Stroage_Dict[layer_Name_List[0]]), 0, SD), 0, np.inf) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Activation_Send:
                        connection_Key = self.Extract_Connection(layer_Name_List[0], layer_Name_List[1]);
                        
                        damage_Type, SD = process["Connection_Control_Dict"][connection_Key];
                        if damage_Type == Damage_Type.On:
                            if not layer_Name_List[1] in layer_Activation_Stroage_Dict.keys():
                                layer_Activation_Stroage_Dict[layer_Name_List[1]] = tf.matmul(layer_Activation_Dict[layer_Name_List[0]], self.weightMatrix_Dict[connection_Key]);     
                            else:
                                layer_Activation_Stroage_Dict[layer_Name_List[1]] += tf.matmul(layer_Activation_Dict[layer_Name_List[0]], self.weightMatrix_Dict[connection_Key]);                                
                        elif damage_Type == Damage_Type.Off:
                            if not layer_Name_List[1] in layer_Activation_Stroage_Dict.keys():   #If code is just 'continue', it may make a exception when model calcuate activation.
                                layer_Activation_Stroage_Dict[layer_Name_List[1]] = tf.matmul(layer_Activation_Dict[layer_Name_List[0]], tf.zeros(tf.shape(self.weightMatrix_Dict[connection_Key])));     
                            else:
                                continue;
                        elif damage_Type == Damage_Type.Damaged:
                            if not layer_Name_List[1] in layer_Activation_Stroage_Dict.keys():
                                layer_Activation_Stroage_Dict[layer_Name_List[1]] = tf.matmul(layer_Activation_Dict[layer_Name_List[0]], self.weightMatrix_Dict[connection_Key] + tf.random_normal(tf.shape(self.weightMatrix_Dict[connection_Key]), 0, SD));     
                            else:
                                layer_Activation_Stroage_Dict[layer_Name_List[1]] += tf.matmul(layer_Activation_Dict[layer_Name_List[0]], self.weightMatrix_Dict[connection_Key] + tf.random_normal(tf.shape(self.weightMatrix_Dict[connection_Key]), 0, SD));

                    elif order_Code == Order_Code.Output_Layer_Error_Calculation_Sigmoid:
                        process["PlaceHolder_Dict"][order_Index] = tf.placeholder(tf.float32);
                        layer_Error_Dict[layer_Name_List[0]] = (process["PlaceHolder_Dict"][order_Index] - layer_Activation_Dict[layer_Name_List[0]]) * layer_Activation_Dict[layer_Name_List[0]] * (1 - layer_Activation_Dict[layer_Name_List[0]]) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Output_Layer_Error_Calculation_Softmax:
                        process["PlaceHolder_Dict"][order_Index] = tf.placeholder(tf.float32);
                        layer_Error_Dict[layer_Name_List[0]] = process["PlaceHolder_Dict"][order_Index] - layer_Activation_Dict[layer_Name_List[0]] * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Hidden_Layer_Error_Calculation_Sigmoid:
                        layer_Error_Dict[layer_Name_List[0]] = layer_Error_Stroage_Dict[layer_Name_List[0]] * layer_Activation_Dict[layer_Name_List[0]] * (1 - layer_Activation_Dict[layer_Name_List[0]]) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Hidden_Layer_Error_Calculation_ReLU:
                        layer_Error_Dict[layer_Name_List[0]] = layer_Error_Dict[layer_Name_List[0]] * tf.sign(layer_Activation_Dict[layer_Name_List[0]]) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Error_Send:
                        connection_Name = self.Extract_Connection(layer_Name_List[1], layer_Name_List[0]);

                        damage_Type, SD = process["Connection_Control_Dict"][connection_Name];
                        if damage_Type == Damage_Type.On:
                            if not layer_Name_List[1] in layer_Error_Stroage_Dict.keys():
                                layer_Error_Stroage_Dict[layer_Name_List[1]] = tf.matmul(layer_Error_Dict[layer_Name_List[0]], tf.transpose(self.weightMatrix_Dict[connection_Name]));     
                            else:
                                layer_Error_Stroage_Dict[layer_Name_List[1]] += tf.matmul(layer_Error_Dict[layer_Name_List[0]], tf.transpose(self.weightMatrix_Dict[connection_Name]));                                
                        elif damage_Type == Damage_Type.Off:  #If code is just 'continue', it may make a exception when renew calcuate weight and bias.
                            if not layer_Name_List[0] in layer_Error_Dict.keys():
                                layer_Error_Stroage_Dict[layer_Name_List[1]] = tf.zeros(tf.shape(layer_Activation_Dict[layer_Name_List[1]]));
                            else:
                                continue;
                        elif damage_Type == Damage_Type.Damaged:
                            if not layer_Name_List[0] in layer_Error_Dict.keys():
                                layer_Error_Stroage_Dict[to_Layer_Name] = tf.matmul(layer_Activation_Dict[layer_Name_List[0]], tf.transpose(self.weightMatrix_Dict[connection_Name] + tf.random_normal(tf.shape(self.weightMatrix_Dict[connection_Name]), 0, SD)));     
                            else:
                                layer_Error_Stroage_Dict[to_Layer_Name] += tf.matmul(layer_Activation_Dict[layer_Name_List[0]], tf.transpose(self.weightMatrix_Dict[connection_Name] + tf.random_normal(tf.shape(self.weightMatrix_Dict[connection_Name]), 0, SD)));

                    elif order_Code == Order_Code.Activation_Extract:
                        process["Tensor_List"].append(layer_Activation_Dict[layer_Name_List[0]]);
                        process["Extract_Activation_Tensor_Index_Dict"][order_Index] = len(process["Tensor_List"]) - 1;
                        process["Extract_Activation_Tensor_Cycle_Dict"][order_Index] = current_Cycle;
                        
                    elif order_Code == Order_Code.Bias_Renewal:
                        process["Tensor_List"].append(tf.assign(self.biasMatrix_Dict[layer_Name_List[0]], (self.biasMatrix_Dict[layer_Name_List[0]] + self.config_Variables_Dict["Learning_Rate"] * tf.reduce_sum(layer_Error_Dict[layer_Name_List[0]], 0, True)) * (1-self.config_Variables_Dict["Decay_Rate"])));

                    elif order_Code == Order_Code.Weight_Renewal:
                        from_Layer_Name = self.connection_Information_Dict[connection_Name_List[0]]["From_Layer_Name"];
                        to_Layer_Name = self.connection_Information_Dict[connection_Name_List[0]]["To_Layer_Name"];
                        process["Tensor_List"].append(tf.assign(self.weightMatrix_Dict[connection_Name_List[0]], (self.weightMatrix_Dict[connection_Name_List[0]] + self.config_Variables_Dict["Learning_Rate"] * tf.matmul(tf.transpose(layer_Activation_Dict[from_Layer_Name]),layer_Error_Dict[to_Layer_Name])) * (1-self.config_Variables_Dict["Decay_Rate"])));

                    elif order_Code == Order_Code.Layer_Duplication:                        
                        if layer_Name_List[0] in layer_Activation_Stroage_Dict.keys():
                            layer_Activation_Stroage_Dict[layer_Name_List[1]] = layer_Activation_Stroage_Dict[layer_Name_List[0]];
                        if layer_Name_List[0] in layer_Activation_Dict.keys():
                            layer_Activation_Dict[layer_Name_List[1]] = layer_Activation_Dict[layer_Name_List[0]];
                        if layer_Name_List[0] in layer_Error_Stroage_Dict.keys():
                            layer_Error_Stroage_Dict[layer_Name_List[1]] = layer_Error_Stroage_Dict[layer_Name_List[0]];
                        if layer_Name_List[0] in layer_Error_Dict.keys():
                            layer_Error_Dict[layer_Name_List[1]] = layer_Error_Dict[layer_Name_List[0]];

                    elif order_Code == Order_Code.Connection_Duplication:
                        process["Tensor_List"].append(tf.assign(self.weightMatrix_Dict[connection_Name_List[1]], self.weightMatrix_Dict[connection_Name_List[0]]));

                    elif order_Code == Order_Code.Transposed_Connection_Duplication:
                        process["Tensor_List"].append(tf.assign(self.weightMatrix_Dict[connection_Name_List[1]], tf.transpose(self.weightMatrix_Dict[connection_Name_List[0]])));

                    elif order_Code == Order_Code.Bias_Equalization:
                        bias_Sum = tf.zeros(tf.shape(self.biasMatrix_Dict[layer_Name_List[0]]), dtype=tf.float32);
                        for layer_Name in layer_Name_List:
                            bias_Sum += self.biasMatrix_Dict[layer_Name];
                        bias_Average = bias_Sum / len(layer_Name_List);
                        for layer_Name in layer_Name_List:
                            process["Tensor_List"].append(tf.assign(self.biasMatrix_Dict[layer_Name], bias_Average));

                    elif order_Code == Order_Code.Weight_Equalization:
                        connection_Sum = tf.zeros(tf.shape(self.weightMatrix_Dict[connection_Name_List[0]]), dtype=tf.float32);
                        for connection_Name in connection_Name_List:
                            connection_Sum += self.weightMatrix_Dict[connection_Name];
                        connection_Average = connection_Sum / len(connection_Name_List);
                        for connection_Name in connection_Name_List:
                            process["Tensor_List"].append(tf.assign(self.weightMatrix_Dict[connection_Name], connection_Average));

                    elif order_Code == Order_Code.Layer_Initialize:
                        if layer_Name_List[0] in layer_Activation_Stroage_Dict.keys():
                            del layer_Activation_Stroage_Dict[layer_Name_List[0]];
                        if layer_Name_List[0] in layer_Activation_Dict.keys():
                            del layer_Activation_Dict[layer_Name_List[0]];
                        if layer_Name_List[0] in layer_Error_Stroage_Dict.keys():
                            del layer_Error_Stroage_Dict[layer_Name_List[0]];
                        if layer_Name_List[0] in layer_Error_Dict.keys():
                            del layer_Error_Dict[layer_Name_List[0]];

                    elif order_Code == Order_Code.Cycle_Marker:
                        current_Cycle += 1;
                    
                    elif order_Code == Order_Code.Uniform_Random_Activation_Insert:
                        shape = (tf.shape(process["PlaceHolder_Dict"]["Probability_Filter"])[0], self.layer_Information_Dict[layer_Name_List[0]]["Unit"]);
                        damage_Type, SD = process["Layer_Control_Dict"][layer_Name_List[0]];
                        if damage_Type == Damage_Type.On:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(tf.random_uniform(shape, maxval=order_Variable_List[0]) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1), 0, 1);
                        elif damage_Type == Damage_Type.Off:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.zeros(shape);
                        elif damage_Type == Damage_Type.Damaged:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(tf.random_uniform(shape, maxval=order_Variable_List[0]) + tf.random_normal(shape, 0, SD), 0, 1) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.Normal_Random_Activation_Insert:
                        shape = (tf.shape(process["PlaceHolder_Dict"]["Probability_Filter"])[0], self.layer_Information_Dict[layer_Name_List[0]]["Unit"]);
                        damage_Type, SD = process["Layer_Control_Dict"][layer_Name_List[0]];
                        if damage_Type == Damage_Type.On:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(tf.abs(tf.random_normal(shape, 0, order_Variable_List[0])) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1), 0, 1);
                        elif damage_Type == Damage_Type.Off:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.zeros(shape);
                        elif damage_Type == Damage_Type.Damaged:
                            layer_Activation_Dict[layer_Name_List[0]] = tf.clip_by_value(tf.abs(tf.random_normal(shape, 0, order_Variable_List[0])) + tf.random_normal(shape, 0, SD), 0, 1) * process["PlaceHolder_Dict"]["Probability_Filter"] * tf.clip_by_value(process["PlaceHolder_Dict"]["Cycle_Filter"] - current_Cycle, 0, 1);

                    elif order_Code == Order_Code.End_and_Initialize:
                        continue;
    
    def Learn(self):
        for learning_Setup_Index in range(self.current_Learning_Setup_Index, len(self.learning_Setup_List)):
            learning_Setup = self.learning_Setup_List[learning_Setup_Index];
            if self.current_LearningSetup_Epoch == 0:                
                self.Run_Test(learning_Setup);
            for epoch in range(self.current_LearningSetup_Epoch, learning_Setup["Training_Epoch"]):
                self.Run_Training(learning_Setup);
                self.Run_SaveCurrentWeight();
                
                self.current_Total_Epoch += 1;
                self.current_LearningSetup_Epoch += 1;

                if self.current_LearningSetup_Epoch % learning_Setup["Test_Timing"] == 0:
                    self.Run_Test(learning_Setup);

                if self.pause_Status:                    
                    break;

            if self.pause_Status:
                break;
            else:
                self.current_Learning_Setup_Index += 1;
                self.current_LearningSetup_Epoch = 0;
        
        if not self.pause_Status:
            self.Test_Result_Save();    #Auto Save
            self.pause_Status = True;

    def Run_Training(self, learning_Setup):
        #Minibatch Making & Probability filter & Traning Sequence Randomize;
        training_Matching_Index_List = list(range(len(learning_Setup["Training_Matching_List"])));
        if learning_Setup["Shuffle_Mode"] in [Shuffle_Mode.Matching_Random_Pattern_Random, Shuffle_Mode.Matching_Random_Pattern_Sequence]:
            shuffle(training_Matching_Index_List);

        training_Data_List = [];
        for training_Matching_Index in training_Matching_Index_List:                        
            training_Matching_Information = learning_Setup["Training_Matching_List"][training_Matching_Index];                    
            pattern_Pack = self.pattern_Pack_Dict[training_Matching_Information["Pattern_Pack_Name"]];
            process = self.process_Dict[training_Matching_Information["Process_Name"]];
            
            probability_Filter = np.round(pattern_Pack["Probability"] + 0.5 - np.random.rand(pattern_Pack["Count"], 1));

            pattern_Index_List = list(range(pattern_Pack["Count"]));
            if learning_Setup["Shuffle_Mode"] in [Shuffle_Mode.Matching_Random_Pattern_Random, Shuffle_Mode.Matching_Sequence_Pattern_Random]:
                shuffle(pattern_Index_List);

            training_Data_List_at_Single_Matching = [];
            for mini_Batch_Start_Index in range(0, pattern_Pack["Count"], learning_Setup["Mini_Batch_Size"]):
                selected_Pattern_Index_List = pattern_Index_List[mini_Batch_Start_Index:mini_Batch_Start_Index+learning_Setup["Mini_Batch_Size"]];

                feed_Dict = {};
                feed_Dict[process["PlaceHolder_Dict"]["Probability_Filter"]] = probability_Filter[selected_Pattern_Index_List];
                feed_Dict[process["PlaceHolder_Dict"]["Cycle_Filter"]] = pattern_Pack["Cycle"][selected_Pattern_Index_List];

                for order_Index in process["PlaceHolder_Dict"].keys():
                    if order_Index in ["Probability_Filter", "Cycle_Filter"]:
                        continue;
                    feed_Dict[process["PlaceHolder_Dict"][order_Index]] = pattern_Pack[training_Matching_Information["Assign"][order_Index]][selected_Pattern_Index_List];

                training_Data_List_at_Single_Matching.append((process["Tensor_List"], feed_Dict));                
            
            training_Data_List.extend(training_Data_List_at_Single_Matching);

        if learning_Setup["Shuffle_Mode"] == Shuffle_Mode.Random_All:
            shuffle(training_Data_List);

        for tensor_List, feed_Dict in training_Data_List:
            self.tf_Session.run(tensor_List, feed_dict=feed_Dict);

    def Run_Test(self, learning_Setup):
        for test_Matching_Information in learning_Setup["Test_Matching_List"]:                    
            pattern_Pack = self.pattern_Pack_Dict[test_Matching_Information["Pattern_Pack_Name"]];
            process = self.process_Dict[test_Matching_Information["Process_Name"]];

            probability_Filter = np.ones(shape=(pattern_Pack["Count"], 1)); #Test does not use probability filter.

            feed_Dict = {};
            feed_Dict[process["PlaceHolder_Dict"]["Probability_Filter"]] = probability_Filter;
            feed_Dict[process["PlaceHolder_Dict"]["Cycle_Filter"]] = pattern_Pack["Cycle"];
            for order_Index in process["PlaceHolder_Dict"].keys():
                if order_Index in ["Probability_Filter", "Cycle_Filter"]:
                    continue; 
                feed_Dict[process["PlaceHolder_Dict"][order_Index]] = pattern_Pack[test_Matching_Information["Assign"][order_Index]];

            result = self.tf_Session.run(process["Tensor_List"], feed_dict=feed_Dict);
            for order_Index in process["Extract_Activation_Tensor_Index_Dict"]:                
                for target_Pattern_Name, extract_Order_Index, extract_Data_Type  in test_Matching_Information["Extract_Data"]:
                    if order_Index == extract_Order_Index:
                        result_Key = (extract_Data_Type, self.current_Total_Epoch, learning_Setup["Name"], self.current_LearningSetup_Epoch, process["Extract_Activation_Tensor_Cycle_Dict"][order_Index], test_Matching_Information["Pattern_Pack_Name"], target_Pattern_Name, test_Matching_Information["Process_Name"], order_Index);                        
                        result_Tensor_Index = process["Extract_Activation_Tensor_Index_Dict"][order_Index];
                        self.extract_Result_Dict[result_Key] = result[result_Tensor_Index];
    def Run_SaveCurrentWeight(self):
        key_List = [];
        tensor_List = [];
        for key in self.weightMatrix_Dict.keys():
            key_List.append(("Weight", key));
            tensor_List.append(self.weightMatrix_Dict[key]);
        for key in self.biasMatrix_Dict.keys():
            key_List.append(("Bias", key));
            tensor_List.append(self.biasMatrix_Dict[key]);
        weight_Data_List = self.tf_Session.run(tensor_List);

        self.weight_Dict_for_Observation = {};
        for index in range(len(key_List)):
            self.weight_Dict_for_Observation[key_List[index]] = weight_Data_List[index];        

    def Test_Result_Save(self, save_Directory = None):
        if save_Directory is None:        
            save_Directory = time.strftime('%Y%m%d %H%M%S', time.localtime(time.time())) + " Auto Save";
        if not os.path.isdir(save_Directory):
            os.mkdir(save_Directory);
        
        with open(save_Directory + "/Model_Information.txt", "w", encoding="utf8") as save_Stream:
            save_Stream.write(self.Extract_Simulator_Information());
        
        self.Raw_Activation_Save(save_Directory);
        self.Mean_Squared_Error_Save(save_Directory);
        self.Cross_Entropy_Save(save_Directory);
        self.Semantic_Stress_Save(save_Directory);
        self.WeightAndBias_Save(save_Directory + "/Weight_and_Bias.HNet_Model")
    
    def Raw_Activation_Save(self, save_Directory):
        extract_Data_Row_List = [];
        extract_Data_Row_List.append("Total Epoch\tLearning Setup\tEpoch in Learning Setup\tCycle\tPattern Pack\tUisng Process\tOrder Index\tLayer\tName\tProbability\tRaw Activation");

        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Raw_Activation:
                continue;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.extract_Result_Dict[data_Key];
            
            patternPack = self.pattern_Pack_Dict[pattern_Pack_Name];
            process = self.process_Dict[process_Name];

            layer_Name = process["Order_List"][order_Index][1][0];
            name_List = patternPack["Name"];
            probability_List = patternPack["Probability"].ravel();

            for index in range(raw_Data.shape[0]):
                extract_Data_Row_List.append(
                    str(total_Epoch) + "\t" + 
                    learning_Setup_Name + "\t" + 
                    str(learning_Setup_Epoch) + "\t" +
                    str(cycle) + "\t" + 
                    pattern_Pack_Name + "\t" +
                    process_Name + "\t" + 
                    str(order_Index) + "\t" + 
                    layer_Name + "\t" + 
                    name_List[index] + "\t" + 
                    str(probability_List[index]) + "\t" + 
                    "\t".join([str(x) for x in raw_Data[index]]));
        
        with open(save_Directory + "/Raw_Activation.txt", "w", encoding="utf8") as save_Stream:
            save_Stream.write("\n".join(extract_Data_Row_List));        

    def Mean_Squared_Error_Save(self, save_Directory):
        extract_Data_Row_List = [];
        extract_Data_Row_List.append("Total Epoch\tLearning Setup\tEpoch in Learning Setup\tCycle\tPattern Pack\tTarget Pattern\tUisng Process\tOrder Index\tLayer\tName\tProbability\tMean Squared Error");
        
        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.extract_Result_Dict.keys():
            if not extract_Data_Type == Extract_Data_Type.Mean_Squared_Error:
                continue;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.extract_Result_Dict[data_Key];
            
            patternPack = self.pattern_Pack_Dict[pattern_Pack_Name];
            process = self.process_Dict[process_Name];
            
            layer_Name = process["Order_List"][order_Index][1][0];
            name_List = patternPack["Name"];
            probability_List = patternPack["Probability"].ravel();
            mean_Squared_Error = np.sqrt(np.mean((patternPack[pattern_Name] - raw_Data) ** 2, axis=1));             
            
            for index in range(len(mean_Squared_Error)):
                extract_Data_Row_List.append(
                    str(total_Epoch) + "\t" + 
                    learning_Setup_Name + "\t" + 
                    str(learning_Setup_Epoch) + "\t" +
                    str(cycle) + "\t" + 
                    pattern_Pack_Name + "\t" + 
                    pattern_Name + "\t" + 
                    process_Name + "\t" + 
                    str(order_Index) + "\t" + 
                    layer_Name + "\t" + 
                    name_List[index] + "\t" + 
                    str(probability_List[index]) + "\t" + 
                    str(mean_Squared_Error[index]));
        
        with open(save_Directory + "/Mean_Squared_Error.txt", "w", encoding="utf8") as save_Stream:
            save_Stream.write("\n".join(extract_Data_Row_List));
        
    def Cross_Entropy_Save(self, save_Directory):
        extract_Data_Row_List = [];
        extract_Data_Row_List.append("Total Epoch\tLearning Setup\tEpoch in Learning Setup\tCycle\tPattern Pack\tTarget Pattern\tUisng Process\tOrder Index\tLayer\tName\tProbability\tCross Entropy");

        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.extract_Result_Dict.keys():            
            if not extract_Data_Type == Extract_Data_Type.Cross_Entropy:
                continue;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.extract_Result_Dict[data_Key];
            
            patternPack = self.pattern_Pack_Dict[pattern_Pack_Name];
            process = self.process_Dict[process_Name];

            layer_Name = process["Order_List"][order_Index][1][0];
            name_List = patternPack["Name"];
            probability_List = patternPack["Probability"].ravel();
            cross_Entropy = -(np.mean(patternPack[pattern_Name] * np.log(raw_Data) + (1 - patternPack[pattern_Name]) * np.log(1 - raw_Data), axis = 1));

            for index in range(len(cross_Entropy)):
                extract_Data_Row_List.append(
                    str(total_Epoch) + "\t" + 
                    learning_Setup_Name + "\t" + 
                    str(learning_Setup_Epoch) + "\t" +
                    str(cycle) + "\t" + 
                    pattern_Pack_Name + "\t" + 
                    pattern_Name + "\t" + 
                    process_Name + "\t" + 
                    str(order_Index) + "\t" + 
                    layer_Name + "\t" + 
                    name_List[index] + "\t" + 
                    str(probability_List[index]) + "\t" + 
                    str(cross_Entropy[index]));

        with open(save_Directory + "/Cross_Entropy.txt", "w", encoding="utf8") as save_Stream:
            save_Stream.write("\n".join(extract_Data_Row_List));

    def Semantic_Stress_Save(self, save_Directory):
        extract_Data_Row_List = [];
        extract_Data_Row_List.append("Total Epoch\tLearning Setup\tEpoch in Learning Setup\tCycle\tPattern Pack\tUisng Process\tOrder Index\tLayer\tName\tProbability\tSemantic Stress");

        for extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index in self.extract_Result_Dict.keys():            
            if not extract_Data_Type == Extract_Data_Type.Semantic_Stress:
                continue;
            data_Key = (extract_Data_Type, total_Epoch, learning_Setup_Name, learning_Setup_Epoch, cycle, pattern_Pack_Name, pattern_Name, process_Name, order_Index);
            raw_Data = self.extract_Result_Dict[data_Key];
            
            patternPack = self.pattern_Pack_Dict[pattern_Pack_Name];
            process = self.process_Dict[process_Name];

            layer_Name = process["Order_List"][order_Index][1][0];
            name_List = patternPack["Name"];
            probability_List = patternPack["Probability"].ravel();
            semantic_Stress = np.mean(raw_Data * np.log2(raw_Data) + (1 - raw_Data) * np.log2(1 - raw_Data) + 1, axis = 1);

            for index in range(len(semantic_Stress)):
                extract_Data_Row_List.append(
                    str(total_Epoch) + "\t" + 
                    learning_Setup_Name + "\t" + 
                    str(learning_Setup_Epoch) + "\t" +
                    str(cycle) + "\t" +  
                    pattern_Pack_Name + "\t" +
                    process_Name + "\t" + 
                    str(order_Index) + "\t" + 
                    layer_Name + "\t" + 
                    name_List[index] + "\t" + 
                    str(probability_List[index]) + "\t" + 
                    str(semantic_Stress[index]));

        with open(save_Directory + "/Semantic_Stress.txt", "w", encoding="utf8") as save_Stream:
            save_Stream.write("\n".join(extract_Data_Row_List));

    
    def Extract_Simulator_Information(self):
        information_Text_List = [];

        information_Text_List.append("▶Config");
        information_Text_List.append("└Momentum: " + str(self.config_Variables_Dict["Momentum"]));
        information_Text_List.append("└Learning Rate: " + str(self.config_Variables_Dict["Learning_Rate"]));
        information_Text_List.append("└Decay Rate: " + str(self.config_Variables_Dict["Decay_Rate"]));
        information_Text_List.append("└Initial Weight SD: " + str(self.config_Variables_Dict["Initial_Weight_SD"]));
        information_Text_List.append("└Device Mode: " + str(self.config_Variables_Dict["Device_Mode"]).upper());
        information_Text_List.append("");

        information_Text_List.append("▶Layer");
        for key in self.layer_Information_Dict.keys():
            layer_Information = self.layer_Information_Dict[key];
            information_Text_List.append("└" + key + "(" + str(layer_Information["Unit"]) + ")");
        information_Text_List.append("");

        information_Text_List.append("▶Connection");
        for key in self.connection_Information_Dict.keys():
            connection_Information = self.connection_Information_Dict[key];
            information_Text_List.append("└" + key + "(" + str(connection_Information["From_Layer_Name"]) + " → " + str(connection_Information["To_Layer_Name"]) + ")");
        information_Text_List.append("");

        information_Text_List.append("▶Pattern Pack");
        for key in self.pattern_Pack_Dict.keys():
            pattern_Pack_Information = self.pattern_Pack_Dict[key];
            information_Text_List.append("└" + key + "(" + str(pattern_Pack_Information["Count"]) + ")");
            for pattern_Name in pattern_Pack_Information:
                if pattern_Name in ["Name", "Probability", "Cycle", "Count"]:
                    continue;
                information_Text_List.append(" └" + pattern_Name + str(pattern_Pack_Information[pattern_Name].shape));
        information_Text_List.append("");

        information_Text_List.append("▶Process");
        for key in self.process_Dict.keys():
            process_Information = self.process_Dict[key];
            information_Text_List.append("└" + key);
            information_Text_List.append(" └Layer Control Information");
            for layer_Name in process_Information["Layer_Control_Dict"].keys():
                if process_Information["Layer_Control_Dict"][layer_Name][0] == Damage_Type.On:
                    information_Text_List.append("  └" + str(layer_Name) + " → On");
                elif process_Information["Layer_Control_Dict"][layer_Name][0] == Damage_Type.Off:
                    information_Text_List.append("  └" + str(layer_Name) + " → Off");
                elif process_Information["Layer_Control_Dict"][layer_Name][0] == Damage_Type.Damaged:
                    information_Text_List.append("  └" + str(layer_Name) + " → Damaged(" + str(process_Information["Layer_Control_Dict"][layer_Name][1]) + ")");
            information_Text_List.append(" └Connection Control Information");
            for connection_Name in process_Information["Connection_Control_Dict"].keys():
                if process_Information["Connection_Control_Dict"][connection_Name][0] == Damage_Type.On:
                    information_Text_List.append("  └" + str(connection_Name) + " → On");
                elif process_Information["Connection_Control_Dict"][connection_Name][0] == Damage_Type.Off:
                    information_Text_List.append("  └" + str(connection_Name) + " → Off");
                elif process_Information["Connection_Control_Dict"][connection_Name][0] == Damage_Type.Damaged:
                    information_Text_List.append("  └" + str(connection_Name) + " → Damaged(" + str(process_Information["Connection_Control_Dict"][connection_Name][1]) + ")");
            information_Text_List.append(" └Order");
            order_Index = 0;
            for order_Code, layer_Name_List, connection_Name_List, order_Variable_List in process_Information["Order_List"]:
                if not order_Variable_List is None:
                    variable_Text = " Vals:(" + ", ".join([str(x) for x in order_Variable_List]) + ")";
                else:
                    variable_Text = "";                
                if not layer_Name_List is None:                
                    information_Text_List.append("  └" + str(order_Index) + ": " + str(order_Code)[11:] + " (" + ", ".join([layer for layer in layer_Name_List]) + ")" + variable_Text);
                elif not connection_Name_List is None:                
                    information_Text_List.append("  └" + str(order_Index) + ": " + str(order_Code)[11:] + " (" + ", ".join([connection for connection in connection_Name_List]) + ")" + variable_Text);
                else:
                    information_Text_List.append("  └" + str(order_Index) + ": " + str(order_Code)[11:] + variable_Text);            
                order_Index += 1;                
        information_Text_List.append("");

        information_Text_List.append("▶Learning Setup");
        for learning_Setup in self.learning_Setup_List:
            information_Text_List.append("└" + learning_Setup["Name"]);
            information_Text_List.append(" └Training Epoch: " + str(learning_Setup["Training_Epoch"]));
            information_Text_List.append(" └Test Timing: " + str(learning_Setup["Test_Timing"]));
            information_Text_List.append(" └Mini Batch Size: " + str(learning_Setup["Mini_Batch_Size"]));
            information_Text_List.append(" └Shuffle Mode: " + str(learning_Setup["Shuffle_Mode"])[13:]);
            information_Text_List.append(" └Training Matching Information");
            training_Matching_Index = 0;
            for training_Matching in learning_Setup["Training_Matching_List"]:
                information_Text_List.append("  └Matching " + str(training_Matching_Index));
                information_Text_List.append("   └Process: " + training_Matching["Process_Name"]);
                information_Text_List.append("   └Pattern Pack: " + training_Matching["Pattern_Pack_Name"]);
                information_Text_List.append("   └Assign: " + training_Matching["Pattern_Pack_Name"]);
                for order_Index in training_Matching["Assign"].keys():
                    information_Text_List.append("    └" + str(order_Index) + " ← " + training_Matching["Assign"][order_Index]);
                training_Matching_Index += 1;
            information_Text_List.append(" └Test Matching Information");
            test_Matching_Index = 0;
            for test_Matching in learning_Setup["Test_Matching_List"]:
                information_Text_List.append("  └Matching " + str(test_Matching_Index));
                information_Text_List.append("   └Process: " + test_Matching["Process_Name"]);
                information_Text_List.append("   └Pattern Pack: " + test_Matching["Pattern_Pack_Name"]);
                information_Text_List.append("   └Assign: " + test_Matching["Pattern_Pack_Name"]);
                for order_Index in test_Matching["Assign"].keys():
                    information_Text_List.append("    └" + str(order_Index) + " ← " + test_Matching["Assign"][order_Index]);                
                information_Text_List.append("   └Extract Data: " + test_Matching["Pattern_Pack_Name"]);
                for pattern, order_Index, extract_Type in test_Matching["Extract_Data"]:
                    extract_Layer_Name = self.process_Dict[test_Matching["Process_Name"]]["Order_List"][order_Index][1][0];                
                    if extract_Type == Extract_Data_Type.Raw_Activation:
                        information_Text_List.append("    └" + "Raw Activation(L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
                    elif extract_Type == Extract_Data_Type.Mean_Squared_Error:
                        information_Text_List.append("    └" + "Mean Squared Error(T: " + pattern + ", L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
                    elif extract_Type == Extract_Data_Type.Cross_Entropy:
                        information_Text_List.append("    └" + "Cross Entropy(T: " + pattern + ", L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
                    elif extract_Type == Extract_Data_Type.Semantic_Stress:
                        information_Text_List.append("    └" + "Semantic Stress(L: " + extract_Layer_Name + "(" + str(order_Index) + ")" +")");
                test_Matching_Index += 1;

        return "\n".join(information_Text_List);

    def Extract_Connection(self, from_Layer_Name, to_Layer_Name):
        extract_List = [];
        for key in self.connection_Information_Dict.keys():
            if from_Layer_Name == self.connection_Information_Dict[key]["From_Layer_Name"] and to_Layer_Name == self.connection_Information_Dict[key]["To_Layer_Name"]:
                return key;

        return False;

    def Extract_Connection_List(self, from_Layer_Name):
        extract_List = [];
        for key in self.connection_Information_Dict.keys():
            if from_Layer_Name == self.connection_Information_Dict[key]["From_Layer_Name"]:
                extract_List.append((key, self.connection_Information_Dict[key]["To_Layer_Name"]));

        return extract_List;