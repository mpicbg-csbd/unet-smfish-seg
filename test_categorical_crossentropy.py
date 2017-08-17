info = """
A little test to see if our categorical crossentropy loss is correctly implemented.
The loss is at a minimum when the arguments are identical.

Let's try adding some noise / label uncertainty and see how the loss changes
"""
import numpy as np
import unet
from keras.utils.np_utils import to_categorical

catcross = unet.my_categorical_crossentropy()
catcrossold = unet.my_categorical_crossentropy_old()

n_classes = 2
res2 = np.random.randint(n_classes, size=(17,15,13))
res2cat = to_categorical(res2).reshape(res2.shape + (n_classes,))
c1=catcross(res2cat, res2cat)
c2=catcrossold(res2cat, res2cat)
print(c1,c2)

for scale in np.arange(10)*0.1:
	res3cat = res2cat + np.random.rand(*res2cat.shape)*scale
	res3cat /= res3cat.sum(axis=-1, keepdims=True)

	c1=catcross(res2cat, res3cat)
	c2=catcrossold(res2cat, res3cat)
	print(c1,c2)

# model = unet.get_unet_n_pool(3)
# model.summary()

