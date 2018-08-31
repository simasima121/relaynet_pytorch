import numpy as np
from skimage.measure import compare_ssim as ssim

# The values of the conversion by Elsa
values = [0.0, 0.007843138, 0.011764706, 0.015686275, 0.019607844, 0.023529412, 0.02745098]

def one_hot_encode(inp, num_classes):
	h,w = inp.shape
	encoding = np.zeros(( h, w, num_classes))
	for i in range(num_classes):
		encoding[:,:,i] = inp == i
	return encoding

def list_of_labels(label_img, num_classes):
	''' Get one hot encoding from label image '''
	train_labels = one_hot_encode(label_img, num_classes)
	return train_labels

def find_stats(true_labels, pred_labels):
	'''
	Takes true_labels (H,W,Label) and pred_labels (H,W,Label), return array of TP,FP,TN,FN,Acc,Precision,Recall,Dice/F1 score 
	Dice returns 0.0 if there's class should be classified but actually wasn't
	'''
	num_classes = 8
	class_vals = []
	thresh = 0.0001

	for i in range(num_classes):
		# NOTE: FOR MY CLASSES WITH MANY IMAGES - JUST ADD EXTRA DIMENSION TO pred_labels[x,:,:,i]
		# NOTE: if Precision etc are NaN, means there's no information about those classes in this image therefore remove them from analysis of that label

		# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
		TP = np.sum(np.logical_and(pred_labels[:,:,i] == 1, true_labels[:,:,i] == 1))

		# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
		TN = np.sum(np.logical_and(pred_labels[:,:,i] == 0, true_labels[:,:,i] == 0))

		# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
		FP = np.sum(np.logical_and(pred_labels[:,:,i] == 1, true_labels[:,:,i] == 0))

		# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
		FN = np.sum(np.logical_and(pred_labels[:,:,i] == 0, true_labels[:,:,i] == 1))

		# Accuracy - no of correct predictions / all predictions
		Acc = round((TP + TN)/(TP + TN + FP + FN), 3)

		# Precision - number of True Positives divided by the number of True Positives and False Positives,
		# number of positive predictions divided by the total number of positive class values predicted
		Precision = round(TP / (TP + FP),3)

		# Recall - number of True Positives divided by the number of True Positives and the number of False Negatives. 
		# Number of positive predictions divided by the number of positive class values in the test data.
		Recall = round(TP / (TP + FN),3)

		# F1 Score - 2*((precision*recall)/(precision+recall)). 
		# F1 score conveys the balance between the precision and the recall.
		F1 = round(2 * (Precision * Recall) / (Precision + Recall + thresh), 3) # Dice is Same as F1 Score
		# Computing Dice Score
		Dice = round(2 * TP / (2*TP+FP+FN),3) 

		# Dice should return 0 if Precision is low because it means that it classified a value but got it wrong
		# if Dice != F1:
		#   print(Dice, F1)
		#   print('Dice and F1 not equal')
		# else:
		#   F1 = Dice

		# print('Label:',i)
		# print('TP: {}, FP: {}, TN: {}, FN: {}, Class Accuracy: {}, Precision: {}, Recall: {}, Dice: {}'.format(TP,FP,TN,FN,Acc, Precision, Recall, Dice))
		# print()
		# NOTE: ONLY CARE ABOUT DICE SCORE FOR NOW AS IT'S ONLY 1 WITH A HEADER, JUST RETURN THE DICE SCORE
		# class_vals.append((TP,FP,TN,FN,Acc,Precision,Recall,Dice)) # 
		class_vals.append(Dice)

	return class_vals

def thickness_metrics(true_labels, pred_labels): 
	'''
	Takes True and Predicted Labels that are one hot encoded
	returns 
	- list of avg true thickness, 
	- avg pred thickness, 
	- mean_abs_error for of Thickness
	- Mean Squared Error of Thickness
	- SSIM score (similarity score between layers)
	'''
	avg_true_thickness_list = []
	avg_pred_thickness_list = []
	mean_abs_error_list = [] 
	mean_squared_error_list = [] 
	ssim_list = []
	
	N = 600
	error_of_thickness = []
	for i in range(8):
		# NOTE: IF AVERAGE_TIHCKNESS IS NAN, MEANS NOT IN THIS IMAGE
		true_thickness = []
		pred_thickness = []
		class_error = []
		# For each col, find thickness, compare to actual thickness and sum errors 
		for j in range(N):
			true_col = true_labels[:,j,i] # finding number of values in col, go down axis i.e. index of axis
			pred_col = pred_labels[:,j,i] 

			true_width = np.count_nonzero(true_col) # count number of 1s
			pred_width = np.count_nonzero(pred_col)

			# Finding thickness by looking at pred_width - don't need truth because will find error
			if true_width != 0:
				true_thickness.append(pred_width)
			if pred_width != 0:
				pred_thickness.append(pred_width)

			# If true width is not 0 or pred_width is not 0, append them otherwise there's no label for this image
			if true_width != 0 or pred_width != 0:
				abs_error = abs(true_width - pred_width) # error is pred - true
				class_error.append(abs_error)

		avg_true_thickness = np.average(true_thickness)
		avg_pred_thickness = np.average(pred_thickness)
		mean_abs_error = np.average(class_error)
		mean_squared_error = np.average(np.power(class_error,2))
		s = ssim(true_labels[:,:,i], pred_labels[:,:,i])
		#print('Label: {} \nAverage True Thickness: {}' \
		#      '\nAverage Predicted Thickness: {}' \
		#      '\nMean Absolute Error of Thickness: {}'\
		#      '\nMean Squared Error of Thickness: {}'\
		#      '\nSSIM: {}\n'\
		#      .format(i,avg_true_thickness,avg_pred_thickness,mean_abs_error, mean_squared_error, s))
		avg_true_thickness_list.append(avg_true_thickness)
		avg_pred_thickness_list.append(avg_pred_thickness)
		mean_abs_error_list.append(mean_abs_error) 
		mean_squared_error_list.append(mean_squared_error) 
		ssim_list.append(s)

	return avg_true_thickness_list,avg_pred_thickness_list,mean_abs_error_list, mean_squared_error_list, ssim_list