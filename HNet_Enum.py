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
    Activation_Calculation_Sigmoid = 1;
    Activation_Calculation_Softmax = 2;
    Activation_Calculation_ReLU = 3;
    Activation_Send = 4;
    Output_Layer_Error_Calculation_Sigmoid = 5;
    Output_Layer_Error_Calculation_Softmax = 6;
    Hidden_Layer_Error_Calculation_Sigmoid = 7;
    Hidden_Layer_Error_Calculation_ReLU = 8;
    Error_Send = 9;
    Activation_Extract = 10;
    Bias_Renewal = 11;
    Weight_Renewal = 12;
    Layer_Duplication = 13;
    Connection_Duplication = 14;
    Transposed_Connection_Duplication = 15;
    Bias_Equalization = 16;
    Weight_Equalization = 17;    
    Layer_Initialize = 18;
    Cycle_Marker = 19
    Uniform_Random_Activation_Insert = 20;
    Normal_Random_Activation_Insert = 21;
    End_and_Initialize = 22;

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