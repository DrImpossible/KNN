from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve	
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import pickle


def knn_query(X_test,Y_test,X_train,Y_train,distance_type,num_neighbours,classes):
	KNNTree = NearestNeighbors(n_neighbors=num_neighbours, metric=distance_type, n_jobs=-1).fit(X_train)
	indices = np.array(KNNTree.kneighbors(X_test,n_neighbors=1,return_distance=False),dtype=int)
	#print(indices.shape)
	#print(indices)

	y_score = label_binarize(Y_train[indices[:,0]], classes=classes)
	n_classes = y_score.shape[1]
	y_test = label_binarize(Y_test, classes=classes)

	# Compute Precision-Recall and plot curve
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(n_classes):
	    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
	    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
	
	# Compute micro-average ROC curve and ROC area
	precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
	average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")
	#average_recall["micro"] = average_precision_score(y_test, y_score,average="micro")
	#precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
	print(average_precision["micro"])
	print(precision,recall,average_precision)
	return [precision,recall,average_precision]

def extract_features(model,X_train,X_test=None,layername):
	"""
	Code to extract features from layers of the model.
	Inputs: a) model (A keras model file)
			b) X_train -> Training samples
			c) X_test (optional) -> Testing samples
	Outputs: The extracted features of the model
	"""
	get_layer = Model(input=model.input, output=model.get_layer(layername).output)
  	relu1_features_train = get_layer.predict(X_train)
  	if not X_test == None:
  		relu1_features_test = get_layer.predict(X_test)
  		
  	print('Features extracted!')
  	if not X_test == None:
  		return [relu1_features_train,relu1_features_test]
  	return relu1_features_train

#X_train = np.load('X_train_mnist.npy')
#Y_train = np.load('Y_train_mnist.npy')
#X_test = np.load('X_test_mnist.npy')
#Y_test = np.load('Y_test_mnist.npy')

X_train = np.load('X_train_mnist_w2v.npy')
Y_train = np.load('Y_train_mnist_w2v.npy')
X_test = np.load('X_test_mnist_w2v.npy')
Y_test = np.load('Y_test_mnist_w2v.npy')
#with open('../mnist_train_hash.pkl','rb' ) as f:
#    X_train=pickle.load(f)
#    Y_train=pickle.load(f)
#    pass
#
#with open('../mnist_test_hash.pkl','rb' ) as f:
#    X_test=pickle.load(f)
#    Y_test=pickle.load(f)
#    pass
#
#for i in xrange(6):
pre,rec,avg_prec = knn_query(X_test,Y_test,X_train,Y_train,'euclidean',5,[0,1,2,3,4,5,6,7,8,9])
