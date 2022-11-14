import matplotlib.pyplot as plt


def oc_cmp():
    exp1_loss_oc_iter = []
    exp1_loss_oc = []
    exp2_loss_oc_iter = []
    exp2_loss_oc = []
    exp3_loss_oc_iter = []
    exp3_loss_oc = []
    exp4_loss_oc_iter = []
    exp4_loss_oc = []
    i = 0
    with open("./data/xvitstr_tiny_exp1_loss_fq.txt") as f:
        for line in f:
            if i % 2000 == 0:
                data = line.strip().split(" ")
                exp1_loss_oc_iter.append(int(data[0]) + 1)
                exp1_loss_oc.append(round(eval(data[1]), 2))
            i += 1
    i = 0
    with open("./data/xvitstr_tiny_exp2_loss_fq.txt") as f:
        for line in f:
            if i % 2000 == 0:
                data = line.strip().split(" ")
                exp2_loss_oc_iter.append(int(data[0]) + 1)
                exp2_loss_oc.append(round(eval(data[1]), 2))
            i += 1
    i = 0
    with open("./data/xvitstr_tiny_exp3_loss_fq.txt") as f:
        for line in f:
            if i % 2000 == 0:
                data = line.strip().split(" ")
                exp3_loss_oc_iter.append(int(data[0]) + 1)
                exp3_loss_oc.append(round(eval(data[1]), 2))
            i += 1
    i = 0
    with open("./data/xvitstr_tiny_exp4_loss_fq.txt") as f:
        for line in f:
            if i % 2000 == 0:
                data = line.strip().split(" ")
                exp4_loss_oc_iter.append(int(data[0]) + 1)
                exp4_loss_oc.append(round(eval(data[1]), 2))
            i += 1
    # 显示中文：宋体，6号
    plt.rcParams['font.sans-serif'] = [u'SimSun']
    plt.rcParams['font.size'] = 7.5
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.plot(exp1_loss_oc_iter, exp1_loss_oc, '-y')
    plt.plot(exp2_loss_oc_iter, exp2_loss_oc, '-b')
    plt.plot(exp3_loss_oc_iter, exp3_loss_oc, '-r')
    plt.plot(exp4_loss_oc_iter, exp4_loss_oc, '-g')
    plt.xlabel('迭代次数')
    plt.ylabel('FQ', fontdict={'size':7.5, 'family': 'Times New Roman', 'style': 'italic'})
    plt.xticks(fontproperties='Times New Roman', size=7.5)
    plt.yticks(fontproperties='Times New Roman', size=7.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(['实验$1$', '实验$2$', '实验$3$', '实验$4$'], loc=1, fontsize=7.5)
    plt.savefig("./figures/exp1_exp2_exp3_exp4_loss_fq_300000.jpg",
                dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    oc_cmp()
