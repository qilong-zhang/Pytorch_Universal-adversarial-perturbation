import numpy as np
from deepfool import deepfool
from torch.utils.data import DataLoader
import torch
import os



def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def universal_perturbation(dataset, valset, f, delta=0.02, max_iter_uni = np.inf, xi=8, p=2, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    print('p =', p)
    v = 0
    fooling_rate = 0.0
    best_fooling = 0.0
    num_images =  len(valset) # The images should be stacked ALONG FIRST DIMENSION

    itr = 0

    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        data_loader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory=True)

        print ('Starting pass number ', itr)

        # Go through the data set and compute the perturbation increments sequentially
        k = 0
        f.cuda()
        for cur_img in data_loader:
            k += 1
            # cur_img *= 255.0
            cur_img = cur_img.cuda()
            per = (cur_img + torch.tensor(v).cuda()).cuda()
            if int(f(cur_img).argmax()) == int(f(per).argmax()):
                print('>> k = ', k, ', pass #', itr)

                # Compute adversarial perturbation
                f.zero_grad()
                dr,iter = deepfool(per,
                                   f,
                                   num_classes=num_classes,
                                   overshoot=overshoot,
                                   max_iter=max_iter_df)

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    v = proj_lp(v, xi, p)

        itr = itr + 1
        print('L2 norm = ', np.linalg.norm(v))
        print('abs v max = ', abs(v).flatten().max())
        est_labels_orig = torch.zeros((num_images)).cuda()
        est_labels_pert = torch.zeros((num_images)).cuda()

        batch_size = 100

        # Compute the estimated labels in batches
        test_loader = DataLoader(valset, batch_size = 100, shuffle = False, pin_memory=True, num_workers=8)
        ii = 0
        for img_batch in test_loader:
            if ii % 100 == 0:
                print('-------ii---------- :', ii)
            m = (ii * batch_size)
            M = min((ii + 1) * batch_size, num_images)
            img_batch = img_batch.cuda()
            per_img_batch = (img_batch + torch.tensor(v).cuda()).cuda()
            ii += 1
            est_labels_orig[m:M] = torch.argmax(f(img_batch), dim=1)
            est_labels_pert[m:M] = torch.argmax(f(per_img_batch), dim=1)

        # Compute the fooling rate
        fooling_rate = torch.sum(est_labels_pert != est_labels_orig).float() / num_images
        print('FOOLING RATE = ', fooling_rate)
        if fooling_rate > best_fooling:
            best_fooling = fooling_rate
        print('Best Fooling Rate = ', best_fooling)
        pertbation_name = os.path.join('data', 'P8-{:.2f}-{:2f}.npy'.format(np.linalg.norm(v), fooling_rate*100))
        np.save(pertbation_name, v)

    return v