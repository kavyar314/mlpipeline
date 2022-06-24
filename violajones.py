import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# Next two classes adapted from https://github.com/salvacarrion/viola-jones
class Rectangle:
    '''
    rectangle class -- can compute the value of that rectangle 
    superimposed onto integral images based on the x, y coordinate 
    it should be placed at and width and height


    Attributes:
        x: x coordinate of placement of rectangle
        y: y coordinate of placement of rectangle
        w: width of rectangle (added to the right of the given x coordinate)
        h: height of rectangle (added below the given y coordinate)
    
    Methods:
        compute_area(integral_img)
            Computes the area inside the given rectangle using the integral image
    '''
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
    def compute_area(self, integral_img):
        '''
        integral_img is (n, dim, dim)

        may wish to check all locations for being within bounds?
        '''
        a = (self.x-1, self.y-1)
        b = (self.x+self.w-1, self.y)
        c = (self.x, self.y+self.h-1)
        d = (self.x + self.w-1, self.y+self.h-1)
        n, sh1, sh2 = integral_img.shape
        if b[0] > sh2 or d[1] > sh1:
            print("Index Out of Bounds")
            return None
        return integral_img[:,d[1], d[0]] - integral_img[:,b[1], b[0]] - integral_img[:,c[1], c[0]] + integral_img[:,a[1], a[0]]
    
class HaarFeature:
    '''
    class that computes a Haar Feature

    Attributes:
        positives: rectangles whose pixels should be added to the final value
        negatives: rectangles whose pixels should be subtracted from the final value
    
    Methods:
        compute_value: computes the value of that feature in the passed in integral image
    '''
    def __init__(self, pos, neg):
        self.positives = pos
        self.negatives = neg
    
    def compute_value(self, integral_img):
        '''
        computes value of the feature on the integral img passed in

        Argument:
            integral_img: an array consisting of integral images for the images under consideration
        '''
        positive_amt = np.sum(np.array([r.compute_area(integral_img) for r in self.positives]), axis=0)
        negative_amt = np.sum(np.array([r.compute_area(integral_img) for r in self.negatives]), axis=0)
        return positive_amt - negative_amt
        
def compute_integral_img(imgs):
    '''
    computes the integral image from a given set of images using dynamic programming

    Arguments:
        imgs: array of shape (n, d_1, d_2) containing n images of size d_1 x d_2 each for which integral images must be computed

    '''
    n, d_1, d_2 = imgs.shape
    integral_imgs = np.zeros((n, d_1, d_2))
    for i in range(d_1):
        integral_imgs[:, i, 0] = np.sum(imgs[:, :i+1, 0], axis = 1)
    for j in range(d_2):
        integral_imgs[:, 0, j] = np.sum(imgs[:, 0, :j+1], axis = 1)
    for i in range(1,d_1):
        for j in range(1,d_2):
            integral_imgs[:, i, j] = imgs[:, i, j] + integral_imgs[:, i, j-1] + integral_imgs[:, i-1, j] - integral_imgs[:, i-1, j-1]
    return integral_imgs

def feature_generator(img_dim, base_size=(5,5), scale_stride=1, stride=1):
    '''
    yeilds sets of 4 features at a time corresponding to the four Haar features in the V-J features starting from a given point at a given scale

    Arguments:
        img_dim: dimensions of the image for which features will be generated
        base_size: size of the starting rectangle for each Haar feature
        scale_stride: the amount by which to additively increase scales considered. should be an integer
        stride: amount by which to move the rectangle over when outputing the features
    '''
    scales = np.arange(1, img_dim//min(base_size), int(scale_stride))
    for x_s in scales:
        for y_s in scales:
            start_pts_x = np.arange(0, img_dim, stride, dtype=int)
            start_pts_y = np.arange(0, img_dim, stride, dtype=int)
            for x in start_pts_x:
                for y in start_pts_y:
                    features_to_yield = []
                    w = x_s * base_size[1]
                    h = y_s * base_size[0]
                    if x + 2*w > img_dim or y + 2*h > img_dim:
                        break
                    else:
                        r1 = Rectangle(x, y, w, h)
                        r2 = Rectangle(x+w, y, w, h)
                        features_to_yield.append(HaarFeature([r2], [r1]))

                        r3 = Rectangle(x, y+h, w, h)
                        features_to_yield.append(HaarFeature([r3], [r1]))

                        r4 = Rectangle(x+w, y+h, w, h)
                        features_to_yield.append(HaarFeature([r2, r3], [r1, r4]))

                    if x+3*w > img_dim:
                        break
                    else:
                        r1 = Rectangle(x, y, w, h)
                        r2 = Rectangle(x+w, y, w, h)
                        r5 = Rectangle(x+2*w, y, w, h)

                        features_to_yield.append(HaarFeature([r2], [r1, r5]))
                    yield features_to_yield


class V_J_weak:
    '''
    description: a weak predictor that uses exactly one Viola-Jones feature to classify the image

    Attributes:
        X: set of integral images
        y: set of labels
        feature: the feature used to make a classification
        T: the threshold past which to consider something a positive sample
        stride: amount to space the features apart on the image
        scale_stride: amount by which to space the scales of features computed
        score: performance of the threshold


    Methods:
        fit: fit a weak classifier using the training images and labels provided
        predict: predict the output on new INTEGRAL images

    '''
    def __init__(self, X, y, ft_stride=5, scale_stride=10):
        '''
        X is the set integral images
        '''
        self.X = X
        self.y = y
        self.feature = None
        self.T = 0
        self.stride = ft_stride
        self.scale_stride = scale_stride
        self.score = 0

    def fit(self, X, y, sample_weight=None):
        '''
        fit a weak classifier using the training images and labels provided

        Arguments:
            X: set of integral images
            y: set of labels
            sample_weight: relative weights of the samples (i.e., relative importance of getting a sample correct); defaults to None, in which case it is uniform.
        '''
        n, img_dim, _ = X.shape
        feature_gen = feature_generator(img_dim, scale_stride=self.scale_stride, stride=self.stride)
        max_score = -np.inf
        best_feature = None
        if sample_weight is None:
            sample_weight = 1/n*np.ones(n)
        i = 0
        while True:
            i = i+1
            try:
                item = next(feature_gen)
            except:
                break
            for f in item:
                features = f.compute_value(X)
                pos_ft, pos_wt = features[y==1], sample_weight[y==1]
                neg_ft, neg_wt = features[y==-1], sample_weight[y==-1]
                T, s = find_and_score_threshold(pos_ft, neg_ft, pos_wt, neg_wt)
                if s > max_score:
                    max_score = s
                    best_feature = (f, T)
        self.feature, self.T = best_feature
        self.score = max_score
        return self
        
    def predict(self, X):
        return 2*(self.feature.compute_value(X) > self.T) - 1
    
    
def find_and_score_threshold(class_1, class_2, wt_pos, wt_neg):
    '''
    finds and scores a single threshold for separating between two classes, weighted by the weighting of the samples.

    Arguments:
        class_1: value of feature for items in class 1
        class_2: value of feature for items in class 2
        wt_pos: weights for the samples in class 1 (i.e., relative importance of getting the samples correct)
        wt_neg: weights for the samples in class 2 
    '''
    min_pt = min(min(class_1), min(class_2))
    max_pt = max(max(class_1), max(class_2))    
    score = lambda T: (np.sum(np.multiply((class_1 > T), wt_pos)) + np.sum(np.multiply((class_2 < T),wt_neg)))/(np.sum(wt_pos) + np.sum(wt_neg))
    potential_ts = np.linspace(min_pt, max_pt, 1000)
    scored_ts = [score(t) for t in potential_ts]
    i = np.argmax(scored_ts)
    return potential_ts[i], scored_ts[i]


def extract_rights_wrongs(boosted_clf, processed_X_test, X_test, y_test, idx=None):
    '''
    extracts the images on which a classifier is correct and wrong. The classifier 
    considered is either the full boosted classifier (if idx = None) or the weak 
    classifier at index idx.
    it takes in processed Xs for inference but also raw Xs to output

    Arguments:
        boosted_clf: boosted classifier. must have .predict implemented and a .learned_clfs attribute that is a list
        processed_X_test: integral images for the test images
        X_test: images of the test images
        y_test: labels of the test images
        idx: index of the weak classifier to consider; defaults to None in which case the full boosted classifier is considered.
    '''
    if idx is not None:
        clf = boosted_clf.learned_clfs[idx]
    else:
        clf = boosted_clf
    pred = clf.predict(processed_X_test)
    wrong_idxs = np.where(np.not_equal(pred, y_test))[0]
    right_idxs = np.where(np.equal(pred, y_test))[0]
    class_1_incorrect = [X_test[i] for i in wrong_idxs if y_test[i]==1]
    class_0_incorrect = [X_test[i] for i in wrong_idxs if y_test[i]==-1]
    class_1_correct = [X_test[i] for i in right_idxs if y_test[i]==1]
    class_0_correct = [X_test[i] for i in right_idxs if y_test[i]==-1]
    return class_1_incorrect, class_0_incorrect, class_1_correct, class_0_correct


def plot_confusion_grid(wrong_faces, wrong_notfaces, right_faces, right_notfaces, title_string):
    '''
    plots a confusion grid of 4 of each of faces identified correctly, not faces identified correctly, 
    faces identified incorrectly, and not faces identified incorrectly

    Arguments:
        wrong_faces: images of faces identified as not faces
        wrong_notfaces: images of not faces identified as faces
        right_faces: images of faces identified as faces
        right_notfaces: images of not faces identified as not faces
    '''
    f, axarr = plt.subplots(5,7, figsize=(14,10))#, gridspec_kw={'hspace': 0.1})
    f.tight_layout()
    for i in tqdm(range(5)):
        for j in range(7):
            try:
                if j < 3:
                    if i < 2:
                        axarr[i,j].imshow(wrong_faces[i*3 + j])# , aspect = "auto")
                    if i > 2:
                        axarr[i, j].imshow(wrong_notfaces[(i-3)*3 + j])
                elif j == 3:
                    axarr[i,j].imshow(np.ones((2,2,3)))# , aspect = "auto")
                elif j > 3:
                    if i < 2:
                        axarr[i,j].imshow(right_faces[i*3 + j-4])
                    if i > 2:
                        axarr[i,j].imshow(right_notfaces[(i-3)*3 + j - 4])
            except:
                axarr[i,j].imshow(np.ones((2,2,3)))
            axarr[i,j].axis('off')
    f.subplots_adjust(hspace=0.0, wspace=0.0)#, right=0.7)
    
    f.suptitle(title_string, y=1.02, size=20)
    axarr[0,1].set_title("GOT WRONG", size=16)
    axarr[0,5].set_title("GOT RIGHT", size=16)
    plt.show()