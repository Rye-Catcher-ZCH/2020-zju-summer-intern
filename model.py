import argparse
import os
import pickle
import sys
import timeit

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.svm import SVC


# 注意配置libsvm时,第一要到python里面去make,然后复制对应的.so到相应目录(https://zhuanlan.zhihu.com/p/66612017)
# 特别注意,复制时三个.py文件都要复制到python3.6下面
# path = "/Users/maitianshouwangzhe/desktop/zju-2020-summer-intern/libsvm-3.24/python"
# sys.path.append(path)
# from svmutil import *

def sigmoid(x):
    """The logistic sigmoid function"""
    return 1 / (1 + np.exp(-x))


def softmax(x, dim=-1):
    """The logistic softmax function"""
    # center data to avoid overflow
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)


def one_hot(label, n_samples, n_classes):
    """Onehot function"""
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label.T] = 1
    return one_hot


class LogisticRegression():
    def __init__(self, data_path, save_path, roc_path, max_iter, batch_size, learning_rate, lamda):
        self.valid_acc_list = []
        self.model = []
        self.data_path = data_path  # 数据存储路径
        self.save_path = save_path  # 模型存储路径
        self.roc_path = roc_path  # roc曲线存储路径
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda = lamda

    def load_data(self):
        """
        加载数据,并划分为训练集、验证集、测试集
        :return:
        """
        print('loading data...')
        f = h5py.File(self.data_path, 'r')
        data = np.array(f['data'])
        label = np.array(f['label'])
        img_id = np.array(range(len(label)))
        np.random.shuffle(img_id)  # 随机打乱
        # 按照 8:1:1划分
        self.train_x = data[img_id[: int(0.8 * len(img_id))], :]
        self.train_y = label[img_id[: int(0.8 * len(img_id))]]
        self.valid_x = data[img_id[int(0.8 * len(img_id)): int(0.9 * len(img_id))], :]
        self.valid_y = label[img_id[int(0.8 * len(img_id)): int(0.9 * len(img_id))]]
        self.test_x = data[img_id[int(0.9 * len(img_id)):], :]
        self.test_y = label[img_id[int(0.9 * len(img_id)):]]
        self.initialize()

    def initialize(self):
        """
        权重初始化
        :return:
        """
        self.w = np.ones(len(self.train_x[0]))
        self.b = 1

    def predict_prob(self, x):
        """
        预测样本x属于某一类的概率
        :param x: 待预测的样本
        :return: x属于某一类的概率
        """
        h_theta = 1 / (1 + np.exp(-np.matmul(self.w, x) - self.b))
        return h_theta

    def get_negative_log_likelyhood_gradient(self, x, y):
        """
        :param x: 样本
        :param y: 样本标签
        :param lamda:
        :return: 损失函数negative_log_likelyhood相对于w和b的梯度
        """
        gradient_w = np.zeros(len(x[0]))
        gradient_b = 0
        for i in range(len(y)):
            h_theta_i = self.predict_prob(x[i])
            gradient_w += x[i] * (h_theta_i - y[i])
            gradient_b += h_theta_i - y[i]
        gradient_w = gradient_w / len(y) + self.lamda * self.w
        gradient_b = gradient_b / len(y) + self.lamda * self.b
        return (gradient_w, gradient_b)

    def gradient_descent(self, gradent, learning_rate):
        """
        :param gradent: w和b在本轮迭代中的梯度
        :param learning_rate: 学习率
        :return:
        """
        self.w -= learning_rate * gradent[0]
        self.b -= learning_rate * gradent[1]

    def train(self):
        for iter_num in range(self.max_iter):
            for epoch in range(int(len(self.train_y) / self.batch_size)):
                x = self.train_x[epoch * self.batch_size: (epoch + 1) * self.batch_size, :]
                y = self.train_y[epoch * self.batch_size: (epoch + 1) * self.batch_size]
                gradient = self.get_negative_log_likelyhood_gradient(x, y)
                self.gradient_descent(gradient, self.learning_rate)
            valid_acc = self.test(self.valid_x, self.valid_y)  # 验证集acc
            self.valid_acc_list.append(valid_acc)
            print(iter_num, valid_acc)
            # if iter_num % 200 == 0:
        self.save_model(self.save_path)  # 存储训练模型
        test_acc = self.test(self.test_x, self.test_y)  # 测试集acc
        print("test", test_acc)

    def test(self, x, y):
        """
        :param x: 验证集或测试集样本
        :param y: 验证集或测试集标签
        :return: 预测结果准确度acc
        """
        y_pre = []
        for i in range(len(y)):
            y_pre.append(int(self.predict_prob(x[i]) > 0.5))
        y_pre = np.array(y_pre)
        result = (y_pre == y).astype(int)
        acc = np.average(result)
        return acc

    def draw_roc(self):
        print('drawing ROC curve...')
        # print("ROC:")
        modelpath = ["saved/LR/model_1000_200_0-2.npy"]
        datapath = ["datasets/hog_feature_85000.h5"]
        label_fig = ["batchsize_100"]
        for index in range(len(modelpath)):
            self.load_data()
            self.load_model(modelpath[index])
            y_pre = []
            for i in range(len(self.test_y)):
                y_pre.append(self.predict_prob(self.test_x[i]))
            # 这里y_pre用list即可
            fpr, tpr, threshold = roc_curve(self.test_y, y_pre)  # 画图的时候要用预测的概率，而不是你的预测的值
            plt.plot(fpr, tpr, label=label_fig[index])  # 颜色会自动设置

        plt.xlabel('false alarm rate')
        plt.ylabel('recall')
        plt.title('ROC curve')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(self.roc_path)

    def save_model(self, saveroot):
        print('saving model...')
        np.save(saveroot, {'w': self.w, 'b': self.b, 'acc': np.array(self.valid_acc_list)})

    def load_model(self, loadroot):
        print('loading model...')
        tmp = np.load(loadroot, allow_pickle=True).item()
        self.w = tmp.get('w')
        self.b = tmp.get('b')
        self.valid_acc_list = tmp.get('acc').tolist()


class SVM():
    def __init__(self, data_path, roc_path, max_iter):
        self.data_path = data_path  # 数据存储路径
        # self.save_path = save_path  # 模型存储路径
        self.roc_path = roc_path  # roc曲线存储路径
        self.max_iter = max_iter

    def load_data(self):
        """
        加载数据,并划分为训练集、验证集、测试集
        :return:
        """
        print('loading data...')
        f = h5py.File(self.data_path, 'r')
        data = np.array(f['data'])
        label = np.array(f['label'])
        img_id = np.array(range(len(label)))
        np.random.shuffle(img_id)  # 随机打乱
        # 按照 8:2划分
        self.train_x = data[img_id[: int(0.8 * len(img_id))], :]
        self.train_y = label[img_id[: int(0.8 * len(img_id))]]
        self.test_x = data[img_id[int(0.8 * len(img_id)):], :]
        self.test_y = label[img_id[int(0.8 * len(img_id)):]]

    def train(self, option):
        if option == "test_c":
            c_range = np.logspace(-5, 0, 6, base=2)
            for c in c_range:
                print("C = " + str(c))
                svc = SVC(C=c, kernel='rbf', probability=True, max_iter=self.max_iter)  # probability=True 可以预测概率
                clf = svc.fit(self.train_x, self.train_y)
                # 计算测试集精度
                score = svc.score(self.test_x, self.test_y)
                print('精度为%s' % score)
                save_path = "saved/SVM/svm_c_" + str(c) + "_rbf.pkl"
                print("model saved in " + save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(clf, f)
        elif option == "test_kernel":
            kernels = ['linear', 'rbf', 'poly']
            for kernel in kernels:
                print("kernel = " + kernel)
                svc = SVC(C=0.5, kernel=kernel, probability=True, max_iter=self.max_iter)  # probability=True 可以预测概率
                clf = svc.fit(self.train_x, self.train_y)
                score = svc.score(self.test_x, self.test_y)
                print('精度为%s' % score)
                save_path = "saved/SVM/svm_c_0.5_" + kernel + ".pkl"
                print("model saved in " + save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(clf, f)
        else:
            print("wrong option")

    def text_save(self, filename, data):
        file = open(filename, 'a')
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')
            s = s.replace("'", '').replace(',', '') + '\n'
            file.write(s)
        file.close()
        print("save successfully")

    def txt2list(self, filename):
        l = []
        f = open(filename, 'r')
        for line in f:
            item = line.rstrip("\n")
            l.append(float(item))
        f.close()
        return l

    def draw_roc(self, option):
        if option == "test_c":
            clf = SVC()
            c_range = np.logspace(-5, 0, 6, base=2)
            for c in c_range:
                model_path = "./saved/SVM/svm_c_" + str(c) + "_rbf.pkl"
                with open(model_path, 'rb') as fr:
                    clf = pickle.load(fr)
                y_pred_svm = clf.predict_proba(self.test_x)[:, 1]  # 预测概率
                fpr, tpr, threshold = roc_curve(self.test_y, y_pred_svm)  # 画图的时候要用预测的概率，而不是你的预测的值
                # print(type(fpr))
                self.text_save("C_" + str(c) + "_rbf_fpr.txt", fpr)
                self.text_save("C_" + str(c) + "_rbf_tpr.txt", tpr)
            for c in c_range:
                print(c)
                rec = self.txt2list("C_" + str(c) + "_rbf_tpr.txt")
                far = self.txt2list("C_" + str(c) + "_rbf_fpr.txt")
                plt.plot(far, rec, label="C = " + str(c))
            plt.xlabel('false alarm rate')
            plt.ylabel('recall')
            plt.title('ROC curve')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(self.roc_path)
            plt.show()
            return
        elif option == "test_kernel":
            kernels = ['linear', 'rbf', 'poly']
            for kernel in kernels:
                model_path = "./saved/SVM/svm_c_0.5_" + kernel + ".pkl"
                with open(model_path, 'rb') as fr:
                    clf = pickle.load(fr)
                y_pred_svm = clf.predict_proba(self.test_y)[:, 1]  # 预测概率
                fpr, tpr, threshold = roc_curve(self.test_y, y_pred_svm)  # 画图的时候要用预测的概率，而不是你的预测的值
                # print(type(fpr))
                self.text_save("C_0.5_" + kernel + "_fpr.txt", fpr)
                self.text_save("C_0.5_" + kernel + "_tpr.txt", tpr)
            for kernel in kernels:
                rec = self.txt2list("C_0.5_" + kernel + "_tpr.txt")
                far = self.txt2list("C_0.5_" + kernel + "_fpr.txt")
                plt.semilogx(far, rec, label="kernel = " + kernel)
            plt.xlabel('false alarm rate')
            plt.ylabel('recall')
            plt.title('ROC curve')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(self.roc_path)
            plt.show()
            return
        else:
            print("wrong option")
            return


# using tanh
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, input=None, W=None, b=None,
                 activation='tanh'):
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=float
            )
            if activation == 'sigmoid':
                W_values *= 4
            self.W = W_values
        else:
            self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=float)
            self.b = b_values
        else:
            self.b = b

        # lin_output = np.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]
        self.output = None

    def forward(self, X):
        lin_output = np.dot(X, self.W) + self.b
        probs_pred = np.tanh(lin_output)
        self.output = probs_pred
        return probs_pred

    def get_output_delta(self, next_W, next_delta):
        derivative = 1 - self.output ** 2
        self.delta = np.dot(next_delta, next_W.transpose()) * derivative

    def update_w_and_b(self, x, learning_rate, L2_lamda):
        delta_w = - 1.0 * np.dot(x.transpose(), self.delta) / x.shape[1]
        # delta_w = - 1.0 * np.dot(x.transpose(), self.delta) / x.shape[0]
        # delta_b = - 1.0 * np.mean(self.delta, axis=0)
        delta_b = - 1.0 * np.mean(self.delta, axis=0) / x.shape[1]
        self.W -= learning_rate * (delta_w + L2_lamda * self.W)
        self.b -= learning_rate * delta_b


# using softmax
class OutputLayer(object):
    def __init__(self, rng, n_in, n_out, input=None, W=None, b=None,
                 activation='softmax'):
        self.input = input
        self.delta = 0
        self.n_in = n_in
        self.n_out = n_out
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=float
            )
            if activation == 'softmax':
                W_values *= 4
            self.W = W_values
        else:
            self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=float)
            self.b = b_values
        else:
            self.b = b

        # lin_output = np.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

        self.output = None

    def forward(self, X):
        lin_output = np.dot(X, self.W) + self.b
        probs = softmax(lin_output)
        self.output = probs
        return probs

    def get_nll(self, y, probs_pred, L2_lamda):
        r"""
        Penalized negative log likelihood of the targets under the current
        model.
        """
        n_samples = probs_pred.shape[0]

        norm_beta = np.linalg.norm(self.W, ord=2)
        # get y_one_hot
        y_one_hot = one_hot(
            label=y,
            n_samples=n_samples,
            n_classes=self.n_out
        )

        nll = -np.sum(y_one_hot * np.log(probs_pred))
        penalty = (L2_lamda / 2) * norm_beta ** 2
        # loss = (penalty + nll) / n_samples
        loss = penalty + (nll / n_samples)
        return loss

    def get_output_delta(self, y):
        probs_pred = self.output
        n_samples = probs_pred.shape[0]
        y_one_hot = one_hot(
            label=y,
            n_samples=n_samples,
            n_classes=self.n_out
        )

        self.delta = y_one_hot - probs_pred

    def update_w_and_b(self, x, learning_rate, L2_lamda):
        # delta_w = - 1.0 * np.dot(x.transpose(), self.delta) / x.shape[0]
        # delta_b = - 1.0 * np.mean(self.delta, axis=0)
        delta_w = -1.0 * np.dot(x.transpose(), self.delta) / x.shape[1]
        delta_b = -1.0 * np.mean(self.delta, axis=0) / x.shape[1]
        self.W -= learning_rate * (delta_w + L2_lamda * self.W)
        self.b -= learning_rate * delta_b


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.input = input
        self.hiddenLayer = []
        self.hidden_layer_number = len(n_hidden)  # number of hidden layers
        n_in_tmp = n_in
        for num in n_hidden:
            hidden_layer = HiddenLayer(
                rng=rng,
                n_in=n_in_tmp,
                n_out=num,
                activation='tanh'
            )
            n_in_tmp = num
            self.hiddenLayer.append(hidden_layer)

        self.outputLayer = OutputLayer(
            rng=rng,
            n_in=n_in_tmp,
            n_out=n_out,
            activation='sigmoid'
        )

    def predict(self, input):
        input_tmp = input
        for hl in self.hiddenLayer:  # 隐藏层计算
            hl.forward(input_tmp)
            input_tmp = hl.output

        self.outputLayer.forward(input_tmp)
        return self.outputLayer.output

    def predict_class(self, X):
        probs = self.predict(X)
        return np.argmax(probs, axis=1).reshape((-1, 1))

    def _errors(self, X, y):
        y_pred = self.predict_class(X)
        return np.mean(y != y_pred)

    def backpropagation(self, x, y, learning_rate, L2_lamda):
        # update w and b in output layer
        self.outputLayer.get_output_delta(y)
        x_input = self.hiddenLayer[-1].output
        self.outputLayer.update_w_and_b(x_input, learning_rate, L2_lamda)
        next_W = self.outputLayer.W
        next_delta = self.outputLayer.delta
        total = self.hidden_layer_number
        while total > 0:  # 如果有隐藏层
            self.hiddenLayer[total - 1].get_output_delta(next_W, next_delta)
            if total == 1:
                x_input = x  # 如果只有一层隐藏层，则本层的输入就是网络的输入
            else:
                x_input = self.hiddenLayer[total - 2].output  # 本层隐藏层的输入是上一层的输出
            self.hiddenLayer[total - 1].update_w_and_b(x_input, learning_rate, L2_lamda)

            next_W = self.hiddenLayer[total - 1].W
            next_delta = self.hiddenLayer[total - 1].delta
            total = total - 1

    def _train_model(self, X, y, lr=0.1, L2_lamda=0):
        probs_pred = self.predict(X)
        loss = self.outputLayer.get_nll(y, probs_pred, L2_lamda)
        self.backpropagation(X, y, lr, L2_lamda)
        return loss

    def _valid_model(self, X, y):
        errors = self._errors(X, y)
        return errors

    def _test_model(self, X, y):
        errors = self._errors(X, y)
        return errors

    def train(self, X, y, saveroot, lr=0.1, n_epochs=1e3, batch_size=200, patience_value=5000):
        """
        Fit the regression coefficients via gradient descent on the negative
        log likelihood.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The binary targets for each of the `N` examples in `X`.
        lr : float
            The gradient descent learning rate. Default is 1e-7.
        max_iter : float
            The maximum number of iterations to run the gradient descent
            solver. Default is 1e7.
        """
        # get trainset & validset & testset
        x_train, x_valid, x_test = X[0], X[1], X[2]
        y_train, y_valid, y_test = y[0], y[1], y[2]
        # get n_batches
        n_train_batches = x_train.shape[0] // batch_size
        n_valid_batches = x_valid.shape[0] // batch_size
        n_test_batches = x_test.shape[0] // batch_size

        # train
        print('... training the model')
        # early-stopping parameters
        patience = patience_value  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)  # 验证集频率
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                # get loss of train minibatch
                minibatch_avg_cost = self._train_model(
                    X=x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                    y=y_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                    lr=lr
                )
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self._valid_model(
                        X=x_valid[i * batch_size:(i + 1) * batch_size],
                        y=y_valid[i * batch_size:(i + 1) * batch_size]
                    ) for i in range(n_valid_batches)]
                    # compute mean valid-loss
                    this_validation_loss = np.mean(validation_losses)
                    # print valid info
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set
                        test_losses = [self._test_model(
                            X=x_test[i * batch_size:(i + 1) * batch_size],
                            y=y_test[i * batch_size:(i + 1) * batch_size]
                        ) for i in range(n_test_batches)]
                        # compute mean test-loss
                        test_score = np.mean(test_losses)
                        # print test info
                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )
                        # save the best model
                        with open(saveroot, 'wb') as f:
                            pickle.dump(self, f)

                if patience <= iter:
                    done_looping = True
                    break

        # print train info
        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print('The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time)))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)