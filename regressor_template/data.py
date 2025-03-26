import numpy as np

def load_data():
    # 从文件导入数据
    datafile = './work/dataset.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    feature_names = [ 'X', 'Y', 'Z', 'FX' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 列数]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原始数据集拆分成训练集和测试集
    # 使用80%的数据做训练，20%的数据做测试，测试集和训练集不能存在交集
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值和最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)
    
    # 记录数据的归一化参数，在预测时对数据进行归一化
    global max_values
    global min_values
    
    max_values = maximums
    min_values = minimums
    
    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - min_values[i]) / (maximums[i] - minimums[i])

    # 划分训练集和测试集
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data, max_values, min_values

def load_one_example(test_data):
    # 从测试集中随机选择一条作为推理数据
    idx = np.random.randint(0, test_data.shape[0])
    idx = -1
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 将数据格式修改为[1,列数-1]
    one_data =  one_data.reshape([1,-1])

    return one_data, label
