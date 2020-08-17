import argparse
import h5py
import numpy as np
from model import LogisticRegression, SVM, MLP


def load_data(data_path):
    """
    加载数据,并划分为训练集、验证集、测试集
    :return:
    """
    print('loading data...')
    f = h5py.File(data_path, 'r')
    data = np.array(f['data'])
    label = np.array(f['label'])
    img_id = np.array(range(len(label)))
    np.random.shuffle(img_id)  # 随机打乱
    # 按照 8:1:1划分
    train_x = data[img_id[: int(0.8 * len(img_id))], :]
    train_y = label[img_id[: int(0.8 * len(img_id))]]
    valid_x = data[img_id[int(0.8 * len(img_id)): int(0.9 * len(img_id))], :]
    valid_y = label[img_id[int(0.8 * len(img_id)): int(0.9 * len(img_id))]]
    test_x = data[img_id[int(0.9 * len(img_id)):], :]
    test_y = label[img_id[int(0.9 * len(img_id)):]]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 任务类型
    parser.add_argument('--task', type=str, default="train")
    # 模型选择
    parser.add_argument('--model', type=str, default="LR")
    # 数据路径
    parser.add_argument('--data_path', type=str, default="datasets/hog_feature_85000.h5")
    # 模型存储路径
    parser.add_argument('--save_path', type=str, default="saved/new_model.npy")
    # roc曲线存储路径
    parser.add_argument('--roc_path', type=str, default="new_roc_pic.png")

    # LR参数
    # 最大迭代次数
    parser.add_argument('--max_iter', type=int, default=1000)
    # batch_size
    parser.add_argument('--batch_size', type=int, default=100)
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=0.2)
    # lamda
    parser.add_argument('--lamda', type=float, default=0)

    # SVM参数
    # option
    parser.add_argument('--option', type=str, default="test_c")
    args = parser.parse_args()

    if args.model == "LR":
        model = LogisticRegression(args.data_path, args.save_path, args.roc_path, args.max_iter, args.batch_size,
                                   args.learning_rate, args.lamda)
        if args.task == "train":
            print('training...')
            model.load_data()
            model.train()
        elif args.task == "roc":
            model.draw_roc()
    elif args.model == "SVM":
        model = SVM(args.data_path, args.roc_path, args.max_iter)
        if args.task == "train":
            print('training...')
            model.load_data()
            model.train(args.option)
        elif args.task == "roc":
            model.load_data()
            model.draw_roc(args.option)
    elif args.model == "MLP":
        if args.task == "train":
            if args.option == "one-layer":
                train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(args.data_path)
                X = (train_x, valid_x, test_x)
                y = (train_y, valid_y, test_y)
                n_in = train_x.shape[1]  # 特征向量的维度
                n_out = 2  # 两个类别
                rng = np.random.RandomState(1234)
                num_list = [[2], [10], [20], [50], [100], [500]]  # 单层网络神经元数量
                for n in num_list:
                    model = MLP(
                        rng=rng,
                        input=X,
                        n_in=n_in,
                        n_hidden=n,
                        n_out=n_out
                    )

                    model.train(
                        X=X,
                        y=y,
                        saveroot="saved/MLP/one_layer_" + str(n[0]) + ".pkl",
                        lr=args.learning_rate,
                        batch_size=args.batch_size
                    )
                    prob_list = model.predict(test_x)
                    print(prob_list)
            elif args.option == "two-layer":
                train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(args.data_path)
                X = (train_x, valid_x, test_x)
                y = (train_y, valid_y, test_y)
                n_in = train_x.shape[1]  # 特征向量的维度
                n_out = 2  # 两个类别
                rng = np.random.RandomState(1234)
                num_list = [[20, 8], [10, 4], [50, 10]]  # 两层神经元数量
                for n in num_list:
                    model = MLP(
                        rng=rng,
                        input=X,
                        n_in=n_in,
                        n_hidden=n,
                        n_out=n_out
                    )

                    model.train(
                        X=X,
                        y=y,
                        saveroot="saved/MLP/two_layer_" + str(n[0]) + ".pkl",
                        lr=args.learning_rate,
                        batch_size=args.batch_size
                    )
                    prob_list = model.predict(test_x)
                    print(prob_list)
        elif args.task == "roc":
            pass
            # TODO
    else:
        print("no model")
