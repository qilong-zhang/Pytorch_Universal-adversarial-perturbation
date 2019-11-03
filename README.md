# Python

A PyTorch implementation of universal attack. Refer to the original [*tensorflow code*](https://github.com/LTS4/universal). <br>
## Usage

### Dataset
[training dataset(Choose 10 imgs for each category)](http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar)<br>
[Validation dataset](http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar)

### Get started

To get started, you should first Change the file path to yours. For example
```python
X = ImagetNet('../universal_attack/data/ILSVRC2012_train/pick_image/', 1000, 10, transforms = transform)
```
If your path is "universal_attack/data/", then you should write
```python
X = ImagetNet('universal_attack/data/', 1000, 10, transforms = transform)
```

After you modify all the path, then you can run the following demo code
```
python search.py
```

## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017

