# EEG 신호와 EOG 신호를 모두 사용하여 테스트
# EEG 신호를 DNN 네트워크를 통해 LS 인지 아닌지를 분류
# DNN 네트워크를 통한 테스트 결과 LS 인 경우 출력
# DNN 네트워크를 통한 테스트 결과 LS 가 아닌 경우 CNN 네트워크로 리테스트
# EOG 신호를 CNN 네트워크를 통해 최종 출력 스테이지 분류
# DNN 과 CNN의 혼합 네트워크

from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random
import os
import numpy as np

tf.set_random_seed(777)  # reproducibility


# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.01  # we can use large learning rate using Batch Normalization
training_epochs = 15
batch_size = 100
keep_prob = 0.7

data_length = 0


num_data = 0
each_stage_num = 13000
s1 = 0
s2 = 0
s3= 0
s4 = 0
rr = 0
ww = 0


# 해당 csv 파일에 존재하는 line의 갯수(데이터의 갯수)
with open('../csv_data/'+'band_pass_filter_amplitude(EEG).csv', 'r') as reader:
    for line in reader:
        data_length += 1

# csv 파일에 존재하는 특징들을 x_data 리스트 변수에 저장(0번 index에는 스테이지 이름, 그 외에는 특징값)
def data_input(x_data, num_data, fields):
    for i in range(len(fields)) :
        if i == 0 :
            x_data[num_data][0] = fields[0]

        else :
            x_data[num_data][i] = float(fields[i])

    return x_data[num_data]

# x_data_all 리스트 변수에 저장되어있는 스테이지 이름을 제외한 값들을 x_data 리스트 변수에 저장
def x_data_split(x_data, x_data_all, x_data_num, x_data_all_num) :
    for i in range(len(x_data[x_data_num])) :
        x_data[x_data_num][i] = x_data_all[x_data_all_num][i+1]

    # print(x_data[x_data_num])
    return x_data[x_data_num]

# 각 스테이지별 분포를 변수에 저장
with open('../csv_data/' + 'band_pass_filter_amplitude(EEG).csv', 'r') as reader:
    for line in reader:
        fields = line.split(',')
        character_num = len(fields)

        if fields[0] == 'S3' :
            s3 += 1

        elif fields[0] == 'S4' :
            s4 += 1

        elif fields[0] == 'S1' :
            s1 += 1

        elif fields[0] == 'S2' :
            s2 += 1

        elif fields[0] == 'RR' :
            rr += 1

        elif fields[0] == 'WW' :
            ww += 1

        else :
            print('stage num input error!!')

print(s1, s2, s3, s4, rr, ww)

w, h = character_num-1, data_length
x_data = [[0 for x in range(w)] for y in range(h)]      # x_data 리스트 변수 초기화(csv 라인별 특징갯수 , 데이터 갯수)

y_data = []

w, h = character_num, data_length;
x_data_all = [[0 for x in range(w)] for y in range(h)]      # x_data 리스트 변수 초기화(csv 라인별 특징갯수 + 스테이지명 , 데이터 갯수)

# s1_num, s2_num, s3_num, s4_num, rr_num, ww_num = 0, 0, 0, 0, 0, 0

# all data input in x_data
with open('../csv_data/' + 'band_pass_filter_amplitude(EEG).csv', 'r') as reader:
    for line in reader:
        fields = line.split(',')

        x_data_all[num_data] = data_input(x_data_all, num_data, fields)
        num_data += 1

random.shuffle(x_data_all)      # x_data를 shuffle
# random.shuffle(x_data_all)

x_data_num = 0

# x_data 와 y_data 를 분리 (x_data_all 의 0번 index 값이 y_data에 append)
for x_data_all_num in range(len(x_data_all)) :

        if x_data_all[x_data_all_num][0] == 'S3' :
            x_data[x_data_num] = x_data_split(x_data, x_data_all, x_data_num, x_data_all_num)
            y_data.append(0)
            x_data_num += 1


        elif x_data_all[x_data_all_num][0] == 'S4' :
            x_data[x_data_num] = x_data_split(x_data, x_data_all, x_data_num, x_data_all_num)
            y_data.append(0)
            x_data_num += 1

        elif x_data_all[x_data_all_num][0] == 'S1' :
            x_data[x_data_num] = x_data_split(x_data, x_data_all, x_data_num, x_data_all_num)
            y_data.append(1)
            x_data_num += 1

        elif x_data_all[x_data_all_num][0] == 'S2' :
            x_data[x_data_num] = x_data_split(x_data, x_data_all, x_data_num, x_data_all_num)
            y_data.append(1)
            x_data_num += 1

        elif x_data_all[x_data_all_num][0] == 'RR' :
            x_data[x_data_num] = x_data_split(x_data, x_data_all, x_data_num, x_data_all_num)
            y_data.append(0)
            x_data_num += 1

        elif x_data_all[x_data_all_num][0] == 'WW' :
            x_data[x_data_num] = x_data_split(x_data, x_data_all, x_data_num, x_data_all_num)
            y_data.append(0)
            x_data_num += 1

        else :
            print('y_data input is error!!')


print(x_data)
print('x_data length is ', len(x_data))
print(y_data)

print(len(x_data))
print(len(y_data))



y_test_ds, y_test_ls, y_test_rr, y_test_ww = 0, 0, 0, 0


# y_test data의 스테이지별 분포 (ds, ls, rr, ww stage)
for i in range(len(y_data)):
    if y_data[i] == 0:
        y_test_ds += 1

    elif y_data[i] == 1:
        y_test_ls += 1

    elif y_data[i] == 2:
        y_test_rr += 1

    elif y_data[i] == 3:
        y_test_ww += 1

    else:
        print('error!!')


print('DS test data num is ', y_test_ds)
print('LS test data num is ', y_test_ls)
print('RR test data num is ', y_test_rr)
print('WW test data num is ', y_test_ww)

# one hot encoding
pre_y_test = y_data
y_test = np_utils.to_categorical(y_data)

# print(y_train[3])

print('character num is ', character_num-1)     # 특징의 갯수 출력
x_placeholder_length = character_num - 1

# place holders
X = tf.placeholder(tf.float32, [None, x_placeholder_length])
Y = tf.placeholder(tf.float32, [None, 2])
train_mode = tf.placeholder(tf.bool, name='train_mode')

# layer output size
hidden_output_size = 512
final_output_size = 2

xavier_init = tf.contrib.layers.xavier_initializer()
bn_params = {
    'is_training': train_mode,
    'decay': 0.9,
    'updates_collections': None
}

# We can build short code using 'arg_scope' to avoid duplicate code
# same function with different arguments
with arg_scope([fully_connected],
               activation_fn=tf.nn.relu,
               weights_initializer=xavier_init,
               biases_initializer=None,
               normalizer_fn=batch_norm,
               normalizer_params=bn_params
               ):
    hidden_layer1 = fully_connected(X, hidden_output_size, scope="h1")
    h1_drop = dropout(hidden_layer1, keep_prob, is_training=train_mode)
    hidden_layer2 = fully_connected(h1_drop, hidden_output_size, scope="h2")
    h2_drop = dropout(hidden_layer2, keep_prob, is_training=train_mode)
    hidden_layer3 = fully_connected(h2_drop, hidden_output_size, scope="h3")
    h3_drop = dropout(hidden_layer3, keep_prob, is_training=train_mode)
    hidden_layer4 = fully_connected(h3_drop, hidden_output_size, scope="h4")
    h4_drop = dropout(hidden_layer4, keep_prob, is_training=train_mode)
    hypothesis = fully_connected(h4_drop, final_output_size, activation_fn=None, scope="hypothesis")


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# batch function of data
def next_batch(x_data, y_data, batch_size, i) :
    batch_xs = x_data[batch_size*i : batch_size*(i+1)]
    batch_ys = y_data[batch_size*i : batch_size*(i+1)]

    return batch_xs, batch_ys

# checkpoint save path of learned model
SAVER_DIR = "two_stage_classify_model(EEG)"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "two_stage_classify_model(EEG)")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)


# Test model and check accuracy
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_batch = int(len(x_data) / batch_size)

final_test_accuracy = 0

# learning result list
origin_data = [[], [], [], []]
predict_data = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
predict_data_percent = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
predict_accuracy = [[], [], [], []]

# list parameter initial
for dd in range(4):
    origin_data[dd] = 0
    predict_accuracy[dd] = 0

    for aa in range(4):
        predict_data[dd][aa] = 0
        predict_data_percent[dd][aa] = 0

y_num = 0

# for i in range(test_batch) :
#     batch_xs_test, batch_ys_test = next_batch(x_test, y_test, batch_size, i)
#     test_accuracy = sess.run(accuracy, feed_dict={X: batch_xs_test, Y: batch_ys_test, train_mode: False})
#     final_test_accuracy += test_accuracy
#     test_prediction = sess.run(prediction, feed_dict={X: batch_xs_test, Y: batch_ys_test, train_mode: False})
#     print(test_prediction)



# for i in range(int(len(x_data)/batch_size)):
#     batch_xs_test, batch_ys_test = next_batch(x_data, y_test, batch_size, i)
#
#     test_prediction = sess.run(accuracy, feed_dict={X: batch_xs_test, Y: batch_ys_test, train_mode: False})
#
#     print(test_prediction)




## EOG data test

imagePath = '/tmp/rr_test.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath = '/tmp/EOG(0319)/output_graph(EOG).pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = '/tmp/EOG(0319)/output_labels(EOG).txt'                                   # 읽어들일 labels 파일 경로


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

image_stage_path = os.listdir('../../../EOG_test(0319)/test/')
image_stage_path_len = len(image_stage_path)

batch_size = 10

modelFullPath = '/tmp//EOG(0319)/output_graph(EOG).pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = '/tmp//EOG(0319)/output_labels(EOG).txt'                                   # 읽어들일 labels 파일 경로

def run_inference_on_image_all_data():
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = [[], [], [], [], [], [], [], [], [], []]


    current_num = 0
    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()

    origin_data = [[], [], [], []]
    predict_data = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    predict_data_percent = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    predict_accuracy = [[], [], [], []]
    for dd in range(4):
        origin_data[dd] = 0
        predict_accuracy[dd] = 0

        for aa in range(4) :
            predict_data[dd][aa] = 0
            predict_data_percent[dd][aa] = 0





    for total in range(int(image_stage_path_len / batch_size)):
        for i in range(batch_size):

            # initialize
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)

            x_final_test = x_data[current_num + i : current_num + i + 1]
            y_final_test = y_test[current_num + i : current_num + i + 1]
            # print(x_test)
            test_prediction = sess.run(prediction, feed_dict={X: x_final_test, Y: y_final_test, train_mode: False})
            # print(test_prediction)

            # 원래 데이터의 스테이징 분류(레이블)
            data = image_stage_path[current_num + i].split('_')
            origin_data_label = data[0]

            if origin_data_label == 'S3' or origin_data_label == 'S4':
                origin_data[0] += 1

            elif origin_data_label == 'S1' or origin_data_label == 'S2':
                origin_data[1] += 1

            elif origin_data_label == 'RR':
                origin_data[2] += 1

            elif origin_data_label == 'WW':
                origin_data[3] += 1

            else:
                print('Exist Error Data!!')

            # EEG signal의 특징을 이용한 DNN 네트워크의 테스트 결과가 LS 인 경우
            if test_prediction == 1 :

                # 데이터의 예상 스테이징 분류
                if origin_data_label == 'S3' or origin_data_label == 'S4':
                    predict_data[0][1] += 1

                elif origin_data_label == 'S1' or origin_data_label == 'S2':
                    predict_data[1][1] += 1

                elif origin_data_label == 'RR':
                    predict_data[2][1] += 1

                elif origin_data_label == 'WW':
                    predict_data[3][1] += 1

                else:
                    print('Exist Error Data!!')

                # EEG signal의 특징을 이용한 DNN 네트워크의 테스트 결과가 LS이며, 바르게 예측했을 경우
                if (origin_data_label == 'S1' or origin_data_label == 'S2') and argmax == 1:
                    predict_accuracy[1] += 1

                else:
                    continue

            # EEG signal의 특징을 이용한 DNN 네트워크의 테스트 결과가 LS 가 아닌 경우
            elif test_prediction == 0 :
                image_data = tf.gfile.FastGFile('../../../EOG_test(0319)/test/' + image_stage_path[current_num + i],
                                                'rb').read()

                # EEG signal을 이용한 CNN 네트워크로 다시 한번 테스트
                with tf.Session() as sess:

                    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    predictions = np.squeeze(predictions)
                    argmax = sess.run(tf.argmax(predictions))

                    # 데이터의 예상 스테이징 분류
                    if origin_data_label == 'S3' or origin_data_label == 'S4':
                        if argmax == 0:
                            predict_data[0][0] += 1

                        elif argmax == 1:
                            predict_data[0][1] += 1

                        elif argmax == 2:
                            predict_data[0][2] += 1

                        elif argmax == 3:
                            predict_data[0][3] += 1

                        else:
                            print("Prediction Lable is not available")

                    elif origin_data_label == 'S1' or origin_data_label == 'S2':
                        if argmax == 0:
                            predict_data[1][0] += 1

                        elif argmax == 1:
                            predict_data[1][1] += 1

                        elif argmax == 2:
                            predict_data[1][2] += 1

                        elif argmax == 3:
                            predict_data[1][3] += 1

                        else:
                            print("Prediction Lable is not available")

                    elif origin_data_label == 'RR':
                        if argmax == 0:
                            predict_data[2][0] += 1

                        elif argmax == 1:
                            predict_data[2][1] += 1

                        elif argmax == 2:
                            predict_data[2][2] += 1

                        elif argmax == 3:
                            predict_data[2][3] += 1

                        else:
                            print("Prediction Lable is not available")

                    elif origin_data_label == 'WW':
                        if argmax == 0:
                            predict_data[3][0] += 1

                        elif argmax == 1:
                            predict_data[3][1] += 1

                        elif argmax == 2:
                            predict_data[3][2] += 1

                        elif argmax == 3:
                            predict_data[3][3] += 1

                        else:
                            print("Prediction Lable is not available")

                    else:
                        print('Exist Error Data!!')

                    if (origin_data_label == 'S3' or origin_data_label == 'S4') and argmax == 0:
                        predict_accuracy[0] += 1

                    elif (origin_data_label == 'S1' or origin_data_label == 'S2') and argmax == 1:
                        predict_accuracy[1] += 1

                    elif origin_data_label == 'RR' and argmax == 2:
                        predict_accuracy[2] += 1

                    elif origin_data_label == 'WW' and argmax == 3:
                        predict_accuracy[3] += 1

                    else:
                        continue

                    top_k = predictions.argsort()[-1:][::-1]  # 가장 높은 확률을 가진 1개의 예측값을 얻는다.
                    f = open(labelsFullPath, 'rb')
                    lines = f.readlines()
                    labels = [str(w).replace("\n", "") for w in lines]
                    for node_id in top_k:
                        human_string = labels[node_id]
                        score = predictions[node_id]

                        print('%s (score = %.5f)' % (human_string, score))

            else :
                print('EEG prediction Error!!')

        current_num += batch_size
        # print(origin_data)
        # print(predict_data)
        each_stage_accuracy, total_accuracy = accuracy_func(origin_data, predict_accuracy)



        print(origin_data)
        print(predict_accuracy)
        print(each_stage_accuracy)
        print(total_accuracy)

    each_stage_num = float(predict_data[0][0] + predict_data[0][1] + predict_data[0][2] + predict_data[0][3])

    print("                    ", '{:^40}'.format('측 정 라 벨'))
    print("     -----------------------------------------------------------------------------")
    print("     |       ", '{:^16}'.format('DS'), '{:^16}'.format('LS'), '{:^16}'.format('R'), '{:^16}'.format('W'))
    print("     | DS", '{:>8}({:^6.2f}%)'.format(predict_data[0][0], predict_data[0][0]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[0][1], predict_data[0][1]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[0][2], predict_data[0][2]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[0][3], predict_data[0][3]/each_stage_num * 100))
    print(" 실제 | LS", '{:>8}({:^6.2f}%)'.format(predict_data[1][0], predict_data[1][0]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[1][1], predict_data[1][1]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[1][2], predict_data[1][2]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[1][3], predict_data[1][3]/each_stage_num * 100))
    print(" 라벨 |  R", '{:>8}({:^6.2f}%)'.format(predict_data[2][0], predict_data[2][0]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[2][1], predict_data[2][1]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[2][2], predict_data[2][2]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[2][3], predict_data[2][3]/each_stage_num * 100))
    print("     |  W", '{:>8}({:^6.2f}%)'.format(predict_data[3][0], predict_data[3][0]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[3][1], predict_data[3][1]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[3][2], predict_data[3][2]/each_stage_num * 100),
          '{:>8}({:^6.2f}%)'.format(predict_data[3][3], predict_data[3][3]/each_stage_num * 100))
    print("     -----------------------------------------------------------------------------")

    answer = labels[top_k[0]]
    return answer




def accuracy_func(origin_data, predict_accuracy) :

    each_stage_accuracy = [[], [], [], []]
    for a in range(len(each_stage_accuracy)) :
        # each_stage_accuracy[a]

        if origin_data[a] == 0 :
            continue
        else :
            each_stage_accuracy[a] = (predict_accuracy[a] / origin_data[a])



    total = origin_data[0] + origin_data[1] + origin_data[2] + origin_data[3]
    accuracy = predict_accuracy[0] + predict_accuracy[1] + predict_accuracy[2] + predict_accuracy[3]
    total_accuracy = accuracy / total

    return each_stage_accuracy, total_accuracy


if __name__ == '__main__':
    run_inference_on_image_all_data()