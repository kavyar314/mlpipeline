import numpy as np

class AdaBoost():
	def __init__(self, base_learner, n_learners, sample_size=None, path=None, **base_learner_kwargs):
		if path is not None:
			init_dict = np.load(path).item()
			self.base_clf = init_dict["base"]
			self.classifiers = init_dict["classifiers"]
			self.classifier_weights = init_dict["alphas"]
		else:
			self.base_clf = base_learner
			self.n_learners = n_learners
			self.base_clf_kwargs = base_learner_kwargs

			self.sample_size = sample_size

			self.classifiers = []
			self.classifier_weights = []

	def fit(self, X, y):

		it = 0
		err = 1
		m = X.shape[0]
		D = 1/m * np.ones((m,))
		while it < self.n_learners and np.isclose(err, 0):

			new_clf = self.base_clf(self.base_clf_kwargs)

			if self.sample_size is None: # new clf has the ability to the weighting itself:
				new_clf.fit(X, y, D)
			else:
				idxs = np.random.choice(m, size=self.sample_size, p=D)

				new_clf.fit(X[idxs], y[idxs])

			eps = empirical_err(y, new_clf.predict(X), D)

			alpha = 1/2 * np.log(1/eps - 1)

			D_norm = np.dot(D, np.exp(-alpha*y*new_clf.predict(X)))

			D = np.multiply(D, np.exp(-alpha*y*new_clf.predict(X)))

			self.classifiers.append(new_clf)
			self.classifier_weights.append(alpha)

	def predict(self, X):
		return np.sum(np.array([self.classifier_weights[i]*self.classifiers[i].predict(X) for i in range(len(self.classifier_weights))]), axis=) #TODO: axis

	def get_save_attributes(self):
		return {"base": self.base_clf, "classifiers": self.classifiers, "alphas": self.classifier_weights}

def empirical_err(y, pred, D):
	return np.dot(D, y==pred)
