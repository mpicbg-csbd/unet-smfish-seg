import 

idea = """
Remove crap from segmented image by only keeping the largest connected component of foreground pixels.
"""

ids,cts = np.unique(seg,return_counts=True)
order = np.argsort(cts)[::-1]
seg_clean = seg==ids[order][1]


idea = """
Try different normalization schemes, including adaptive local histogram equalization.
"""

imglist = []
for d in [2, 4, 8, 16]:
	imglist.append(Expo.equalize_adapthist(grey, [d,d]))

info = """
The goals for today are...
1. Try different learning goals. Try 3-channel? Try tissue vs background?
2. Fix normalization that clips brightest places.
3. Think about better cell segmentation
4. Try increasing the cutoff for membrane to > 0.5
"""

def qsave(img):
    print(img.min(), img.max())                                                                                                                                            
    io.imsave('qsave.tif', img)
    !open qsave.tif

def show_mistakes():
	img,pred = io.imread('training/m293/20150915_timeseries_aldob_apoeb_1_05_MembraneMiddle_predict_.tif')
	mask = np.isnan(pred)
	pred[mask]=0
	mask_1 = pred < 0.05
	mask_2 = (0.05 < pred) & (pred < 0.95)
	mask_3 = 0.95 < pred
	p1 = pred.copy()
	p2 = pred.copy()
	p3 = pred.copy()
	p1[~mask_1]=0
	p2[~mask_2]=0
	p3[~mask_3]=0
	qsave(np.stack([p1,p2,p3]).astype('float16'))
	qsave(mask_2.astype('uint8')*100)
	mix = pred.copy()
	mix[mask_1]=0
	mix[mask_2]=1
	mix[mask_3]=2
	res = np.stack([img,mix*100])
	qsave(res)

	lab = io.imread('/Volumes/Coleman_Pocket/Carine_project/data3/labeled_data_membranes/labels_big/smaller2x/20150915_timeseries_aldob_apoeb_1_05_ground_truth.tif')
	lab[lab==2] = 1
	mask = 0.8 < pred
	show = pred.copy()
	show[mask==lab] = 0
	show[mask > lab] = 0
	show[lab > mask] = 1

	res = np.stack([img/img.max(), show, mask, pred, pred])
	qsave(res)

	img2 = img.copy()
	ma = np.percentile(img, 99)
	img2[img2<ma]=0
	qsave(img2)

result = """
Now it's totally clear that the problem areas are just the regions brighter than the 99th percentile, as we can see from img2 above, 
which was clipped in our normalization. Let's see what happens if we train without any normalization...
"""


problem = """
MY MODELS DON'T CONVERGE [EVERY TIME]
This is really two problems:
1. When it fails to converge, what is happening?
2. Why does this happen to me and Laurent, but not to Martin and Uwe? What is different about our problems?
Secondary Problems: 
1. Why does it happen in streaks? Becuase I vary my hyperparams one-at-a-time?


Hypothesis:
- FALSE! Convergence is a very sensitive function of our hyperparams...
Test
Train the same model multiple times. Convergence is random.
Hypotheses:
1. It depends on initial conditions.
2. It depends on the trajectory through param space, and therefore on sort order of X_train.
But for both causes, it's unclear *how* the learning gets stuck.
Hypothesis: We are losing the gradient info early on in the model [during first epoch, and for ALL SAMPLES] so params don't get updated.
Alternative: We are updating params & moving around in param space, but we're lost in a region without
any good local minima. This is consistent with hypothesis 1, but not with hypothesis 2.

This means we can rule out all of the following as causes, although they may still influence the rate.
- Label noise
- Bad normalization
And label noise is not even really a problem with the hand-labeled data...

Tests:
- We can prevent X_train from randomly sorting at the beginning of every epoch. Then we'll be able to say if
learning depends on initialization or not.
- We can also fix the initialized params somehow, then we'll be able to say if our assumptions are true.
Find a set of params that seems to converge about half to the time. Fix these params and use them for testing.
NOTE: In the Ronneberger paper they use a very particular initialization scheme with weights coming from a centered normal
dist with stdev = sqrt(2/fan_in) where fan_in is the number of incoming neurons.

Test:
I've changed the normalization, to min&max [0,1] imagewise. And the percentage of models that converge has gone way down.
This doesn't rule out the 
Hypothesis
- Our problem is just too hard, and would be easier as a regression of the distance-to-membrane...

"""

from scipy.ndimage import distance_transform_edt
dis = distance_transform_edt(1-lab)
dis[dis>20]=20

hypothesis = """
I want to tell if it's really the normalization which has caused all my current difficulties...
clip to 1,99 percentile | no trouble
stakk to 0,1 | lots of trouble
image to 0,1 | ???
image Expo.equalize_adapthist | ???

"""
