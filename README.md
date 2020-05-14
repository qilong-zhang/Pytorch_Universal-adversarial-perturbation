# Python

A PyTorch implementation of universal adversarial perturbation which is more easy to understand and implement. Refer to the original [*tensorflow code*](https://github.com/LTS4/universal). <br>



## Usage

### Dataset
[training dataset(Choose 10 imgs for each category)](http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar)<br>
[Validation dataset](http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar)<br>

All you need to do is unzip the downloaded file, and the results are as follows:

![data list](C:\Users\MSI\Desktop\Pytorch_Universal-attack\data list.png)

![img list](C:\Users\MSI\Desktop\Pytorch_Universal-attack\img list.png)

### Get started

To get started, you should first Change the file path to yours. For example
```python
X = ImagetNet('../data/ILSVRC2012_train/pick_image/', 1000, 10, transforms = transform)
```
If your path is `testing_data_path`, then you should write
```python
X = torch.utils.data.DataLoader(ImageFolder(testing_data_path, transforms = transform)
```

After you modify all the path, then you can run the following demo code
```
python search.py
```

### Result

I tested our code for `googlenet` , and at $\epsilon=10$, our accuracy on the validation set was ~78%.  It is similar with the result report in paper.

![paper result](C:\Users\MSI\Desktop\Pytorch_Universal-attack\paper result.png)

## Reference

[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017

