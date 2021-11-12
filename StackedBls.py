import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time
from utils.utils import *


def BLS(train_x, train_y, test_x, test_y, s, c, stack_params):
    stack_layer_num = len(stack_params)

    L = 0
    train_x = preprocessing.scale(train_x, axis=1)

    ymin = 0
    ymax = 1
    train_acc_all = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始

    Beta1OfEachWindow_stack_list = []
    minOfEachWindow_stack_list = []
    distOfMaxAndMin_stack_list = []
    weightOfEnhanceLayer_stack_list = []
    parameterOfShrink_stack_list = []
    OutputWeight_stack_list = []


    train_out_put = 0.0
    test_out_put = 0.0
    train_y_value = train_y.copy()
    print("-----------TRAIN--------------")
    for k in range(stack_layer_num):
        print("stack step {}".format(k))
        Beta1OfEachWindow = []
        distOfMaxAndMin = []
        minOfEachWindow = []
        N1 = stack_params[k][0]
        N2 = stack_params[k][1]
        N3 = stack_params[k][2]

        FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
        OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])

        for i in range(N2):
            #random.seed(i)
            weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
            FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
            FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
            betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
            Beta1OfEachWindow.append(betaOfEachWindow)
            outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
            #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
            distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
            minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
            outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
            OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
            del outputOfEachWindow
            del FeatureOfEachWindow
            del weightOfEachWindow

        Beta1OfEachWindow_stack_list.append(Beta1OfEachWindow.copy())
        minOfEachWindow_stack_list.append(minOfEachWindow.copy())
        distOfMaxAndMin_stack_list.append(distOfMaxAndMin.copy())

        InputOfEnhanceLayerWithBias = np.hstack(
            [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

        if N1 * N2 >= N3:
            #random.seed(67797325)
            weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
        else:
            #random.seed(67797325)
            weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

        weightOfEnhanceLayer_stack_list.append(weightOfEnhanceLayer.copy())

        tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
        #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

        parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
        parameterOfShrink_stack_list.append(parameterOfShrink.copy())

        OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

        # 生成最终输入
        InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
        pinvOfInput = pinv(InputOfOutputLayer, c)
        OutputWeight = np.dot(pinvOfInput, train_y)
        time_end = time.time()
        trainTime = time_end - time_start

        OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
        OutputWeight_stack_list.append(OutputWeight.copy())

        train_out_put += OutputOfTrain
        train_x = OutputOfTrain

        trainAcc = show_accuracy(train_out_put, train_y_value)
        print('Training accurate is', trainAcc * 100, '%')
        print('Training time is ', trainTime, 's')
        train_acc_all[0][0] = trainAcc
        train_time[0][0] = trainTime

        train_y -= train_x

    # 测试过程
    test_x = preprocessing.scale(test_x, axis=1)

    time_start = time.time()

    print("---------------TEST-------------------")


    test_y_value = test_y.copy()
    test_x_out_put = 0.
    one_stack_acc = 0.
    for k in range(stack_layer_num):
        N1 = stack_params[k][0]
        N2 = stack_params[k][1]
        N3 = stack_params[k][2]

        FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])

        for i in range(N2):
            outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow_stack_list[k][i])
            OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow_stack_list[k][i]) / distOfMaxAndMin_stack_list[k][i] - ymin

        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
        tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer_stack_list[k])

        OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink_stack_list[k])

        InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

        OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight_stack_list[k])

        test_x_out_put = test_x_out_put + OutputOfTest

        one_stack_acc = show_accuracy(test_y_value, test_x_out_put)
        print("{} stack test accurate is {} %".format(k, one_stack_acc * 100))
        test_y -= OutputOfTest


        test_x = OutputOfTest

    time_end = time.time()
    testTime = time_end - time_start
    print('Final Testing accurate is', one_stack_acc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = one_stack_acc
    test_time[0][0] = testTime

    return test_acc, test_time, train_acc_all, train_time


def BLS_AddEnhanceNodes(train_x, train_y, test_x, test_y, s, c, stack_params, L):
    # 生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0


    stack_layer_num = len(stack_params)

    train_acc = np.zeros([1, L + 1])
    train_x = preprocessing.scale(train_x, axis=1)
    test_x = preprocessing.scale(test_x, axis=1)

    ymin = 0
    ymax = 1
    train_acc_all = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始


    train_out_put = 0.0
    test_out_put = 0.0
    train_y_value = train_y.copy()
    test_y_value = test_y.copy()

    print("-----------TRAIN--------------")



    for k in range(stack_layer_num):
        print("stack step {}".format(k))
        Beta1OfEachWindow = []
        distOfMaxAndMin = []
        minOfEachWindow = []
        N1 = stack_params[k][0]
        N2 = stack_params[k][1]
        N3 = stack_params[k][2]

        FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
        OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])



        for i in range(N2):
            #random.seed(i)
            weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
            FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
            FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
            betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
            Beta1OfEachWindow.append(betaOfEachWindow)
            outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
            #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
            distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
            minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
            outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
            OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
            del outputOfEachWindow
            del FeatureOfEachWindow
            del weightOfEachWindow


        #Enhance
        InputOfEnhanceLayerWithBias = np.hstack(
            [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])


        if N1 * N2 >= N3:
            #random.seed(67797325)
            weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
        else:
            #random.seed(67797325)
            weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T


        tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
        #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

        parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)

        OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

        # 生成最终输入
        InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])


        pinvOfInput = pinv(InputOfOutputLayer, c)


        OutputWeight = np.dot(pinvOfInput, train_y)

        OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
        trainAcc = show_accuracy(OutputOfTrain + train_out_put, train_y_value)
        print('Training accurate is', trainAcc * 100, '%')

        #test

        FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])

        for i in range(N2):
            outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
            OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (outputOfEachWindowTest - minOfEachWindow[i]) / \
                                                                      distOfMaxAndMin[i]

        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
        tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

        OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

        InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

        OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
        testAcc = show_accuracy(test_out_put + OutputOfTest, test_y_value)
        print('Testing accurate is', testAcc * 100, '%')

        parameterOfShrinkAdd = []
        M = stack_params[k][3]
        L = stack_params[k][5]
        for e in list(range(L)):
            if e == 0:
                M += stack_params[k][4]
            time_start = time.time()
            if N1 * N2 >= M:
                # random.seed(e)
                weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M) - 1)
            else:
                # random.seed(e)
                weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M).T - 1).T

            tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
            parameterOfShrinkAdd.append(s / np.max(tempOfOutputOfEnhanceLayerAdd))
            OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd * parameterOfShrinkAdd[e])
            tempOfLastLayerInput = np.hstack([InputOfOutputLayer, OutputOfEnhanceLayerAdd])

            D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
            C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
            if C.all() == 0:
                w = D.shape[1]
                B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))
            else:
                B = pinv(C, c)
            pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
            OutputWeight = pinvOfInput.dot(train_y)
            InputOfOutputLayer = tempOfLastLayerInput
            # Training_time = time.time() - time_start
            # train_time[0][e + 1] = Training_time
            OutputOfEnhanceLayerAddTest = tansig(
                InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
            InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest]) #测试用


        OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)



        train_out_put += OutputOfTrain
        train_x = OutputOfTrain

        trainAcc = show_accuracy(train_out_put, train_y_value)
        print('Incremental Training accurate is', trainAcc * 100, '%')

        train_y = train_y - train_x


        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeight)
        test_out_put += OutputOfTest1
        TestingAcc = show_accuracy(test_out_put, test_y_value)
        test_x = OutputOfTest1
        test_y = test_y - test_x

        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %')