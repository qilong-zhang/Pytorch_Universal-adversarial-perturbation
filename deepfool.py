import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, num_classes = 10, overshoot = 0.02, max_iter = 50):
    """
    :param image: Image of size 3*H*W
    :param net: network (input: images, output: values of activation **BEFORE** softmax).
    :param num_class:
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter:
    :return:minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    # net.zero_grad()
    f_image = net(Variable(image, requires_grad = True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1] # 从小到大排序, 再从后往前复制一遍，So相当于从大到小排序
    I = I[0:num_classes] # 挑最大的num_classes个(从0开始，到num_classes结束)
    label = I[0] # 最大的判断的分类

    input_shape = image.detach().cpu().numpy().shape # 原始照片
    pert_image = copy.deepcopy(image) # 干扰照片
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image, requires_grad = True)
    # net.zero_grad()
    fs = net(x)
    k_i = label

    while k_i == label and loop_i < max_iter: # 直到分错类别或者达到循环上限次数
        pert = np.inf
        fs[0,I[0]].backward(retain_graph=True) # x产生了grad
        grad_orig = x.grad.data.cpu().numpy().copy() # original grad
        for k in range(1, num_classes):
            zero_gradients(x) # 将梯度置为0
            fs[0,I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy() # current grad(分类为k(不是目前所划分的那一类)的grad

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0,I[k]] - fs[0,I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert: # 要找到最小的pert
                pert = pert_k
                w = w_k


        # compute r_i and r_tot
        r_i = (pert + 1e-4) * w / np.linalg.norm(w) # 这一次迭代的r
        r_tot = np.float32(r_tot + r_i) #r_total
        pert_image = image + (1+overshoot) * torch.from_numpy(r_tot).cuda()

        x = Variable(pert_image, requires_grad = True) # 将添加干扰后的图片输入net里面
        fs = net(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten()) # 选择最大的一个作为新的分类
        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i