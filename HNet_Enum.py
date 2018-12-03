###############################################################################
# This file is a part of HNet with Tensorflow Core
#
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

from enum import Enum;

class Order_Code(Enum):
    Input_Layer_Acitvation_Insert = 0;
    Activation_Calculation_Linear = 1;
    Activation_Calculation_Sigmoid = 2;
    Activation_Calculation_Softmax = 3;
    Activation_Calculation_Tanh = 4;
    Activation_Calculation_ReLU = 5;
    Activation_Send = 6;
    Output_Layer_Error_Calculation_Linear = 7;
    Output_Layer_Error_Calculation_Sigmoid = 8;
    Output_Layer_Error_Calculation_Softmax = 9;
    Output_Layer_Error_Calculation_Tanh = 10;
    Hidden_Layer_Error_Calculation_Linear = 11;
    Hidden_Layer_Error_Calculation_Sigmoid = 12;
    Hidden_Layer_Error_Calculation_Tanh = 13;
    Hidden_Layer_Error_Calculation_ReLU = 14;
    Error_Send = 15;
    Activation_Extract = 16;
    Bias_Renewal = 17;
    Weight_Renewal = 18;
    Layer_Duplication = 19;
    Connection_Duplication = 20;
    Transposed_Connection_Duplication = 21;
    Bias_Equalization = 22;
    Weight_Equalization = 23;    
    Layer_Initialize = 24;
    Cycle_Marker = 25;
    Uniform_Random_Activation_Insert = 26;
    Normal_Random_Activation_Insert = 27;
    End_and_Initialize = 28;

class Damage_Type(Enum):
    On = 0;
    Off = 1;
    Damaged = 2;

class Shuffle_Mode(Enum):
    Random_All = 0;
    Matching_Random_Pattern_Random = 1;
    Matching_Random_Pattern_Sequence = 2;
    Matching_Sequence_Pattern_Random = 3;
    Matching_Sequence_Pattern_Sequence = 4;

class Extract_Data_Type(Enum):
    Raw_Activation = 0;
    Mean_Squared_Error = 1;
    Cross_Entropy = 2;
    Semantic_Stress = 3;