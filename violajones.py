# Next two classes adapted from https://github.com/salvacarrion/viola-jones
class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
    def compute_area(self, integral_img):
        '''
        integral_img is (n, dim, dim)
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
    def __init__(self, pos, neg):
        self.positives = pos
        self.negatives = neg
    
    def compute_value(self, integral_img):
        positive_amt = np.sum(np.array([r.compute_area(integral_img) for r in self.positives]), axis=0)
        negative_amt = np.sum(np.array([r.compute_area(integral_img) for r in self.negatives]), axis=0)
        return positive_amt - negative_amt
        
def compute_integral_img(imgs):
    '''
    imgs is of shape (n, d_1, d_2)
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
    scales = np.arange(1, img_dim//min(base_size), scale_stride)
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
    min_pt = min(min(class_1), min(class_2))
    max_pt = max(max(class_1), max(class_2))    
    score = lambda T: (np.sum(np.multiply((class_1 > T), wt_pos)) + np.sum(np.multiply((class_2 < T),wt_neg)))/(np.sum(wt_pos) + np.sum(wt_neg))
    potential_ts = np.linspace(min_pt, max_pt, 1000)
    scored_ts = [score(t) for t in potential_ts]
    i = np.argmax(scored_ts)
    return potential_ts[i], scored_ts[i]