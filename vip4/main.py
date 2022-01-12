#!/usr/bin/env python
# coding: utf-8
# authors: Rachel Rea, Tibor Krols, Ninell Oldenburg, Marie Mortensen

#    ### Introduction
#     The goal of the assignment is to implement a prototypical CBIR system. We recommend the use of the CalTech 101 image database http://www.vision.caltech.edu/Image_Datasets/Caltech101/. We recommend that you (for a start) select a subset of say 4-5 categories. When you have checked that everything works you may extend to say 20 categories. For each category, the set of images should be split in two: A training set and a test set (of equal size). The test set must not include images in the training set. When using few categories you may also limit the number of training images (to say 10) per category. Depending on your amount of computational power, for more categories, you may increase the number of training images to the double or more. You should extract visual words using SIFT descriptors (ignoring position, orientation and scale) or similar descriptors extracted at interest points. To compute the descriptors, we recommend to use OpenCV's sift, but other options are possible.

# ### Codebook Generation
#     In order to generate a code book, select a set of training images. Then Extract SIFT features from the training images (ignore position, orientation and scale). The SIFT features should be concatenated into a matrix, one descriptor per row. Then you should run the k-means clustering algorithm on the subset of training descriptors to extract good prototype (visual word) clusters. A reasonable k should be small (say between 200 and 500) for a small number of categories (say 5) and larger (say between 500 and 2000) for a larger number of categories. Also, a good value of k may depend on the complexity of your data. You should experiment with a few di erent values of k (but beware that this can be rather time-consuming). Once clustering has been obtained, classify each training descriptor to the closest cluster centers) and form the bag of words (BoW) for each image in the image training set. Note that there may exist several implementations of k-means available in several libraries, e.g. in OpenCV and in scikit-image. These implementations may dffer both with respect to function, parameters and processing time.

#   ### Indexing
#     The next step consists in content indexing. For each image in the test set you
#     should:
#        
#     a) Extract the SIFT descriptors of the feature points in the image
#     b) Project the descriptors onto the codebook, i.e., for each descriptor the closest cluster prototype should be found
#     c) Construct the generated corresponding bag of words, i.e, word histogram.
# 
#     Please note that you have already performed the same steps for the training images during codebook generation. Now construct and save a table that would contain, per entry at least the file name, the true category, if it belongs to the training- or test set, and the corresponding bag of words / word histogram. The table need only be computed once and then used repeatably in the following retrieval experiments.

# ### Retrieving
#     Finally, you should implement retrieving of images using some of the similarity
#     measures discussed in the course slides. You may use:
#     
#     a) common words
#     b) tf-ifd similarity
#     c) Bhattacharyya distance or Kullback-Leibler divergence
#       
#     Please argue for your choice or report the differences in result when applying the different measures. Your report should show commented results for two experiments. In the first you consider retrieving training images. In the second you test how well you can classify test images. Otherwise the two test are identical. For each test you should count:
#     
#     a) The mean reciprocal rank (i.e. the average across all queries of 1=ranki, where ranki is the rank position of the first correct category for the i'th query).
#     b) How often (in per cent) the correct category is in top-3
# 
#     Please note that the measures above are just two among a long list of possible performance measures. If you google Information retrieval you may  find alternative measures.

#     The report should be kept within 8 pages including everything. About half may show results in form of tables, graphs, and images. Remember to write what the tables and graphs should show. Important decisions, choices, and results should be discussed and explained. In particular, strage/false results should be identified and possible causes should be explained.


# Important packages
import os

import numpy as np
import random

import cv2
import skimage
from skimage.io import imread

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


class Cbir:

    def __init__(self, folder="101_ObjectCategories", categories=5, k_val=200, only_train=False):
        self.folder = folder
        self.categories = categories
        self.k_val = k_val
        self.all_images, self.order, self.all_filenames = self.load_images()
        self.only_train = only_train
        self.desc = 0

        # Empty list for the bag of words / histograms
        self.bow_train = []
        self.bow_test = []

        # Train BoW
        self.train_bow()

    # Loading images from the CALTECH101 corpus
    def load_images(self):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        '''
        Loads images from multiple categories into one array.

        Arguments
        main_folder: folder where subfolders with categories exist
        n_categories: number of categories loaded

        Returns
        final_array: array with dimension (n_categories,) consisting of subarrays with images from each category
        cat: order of categories
        '''

        dr = os.listdir(self.folder)
        cat = dr[0:self.categories]
        all_images = []
        all_filenames = []

        for idx_cat, i in enumerate(cat):
            tmp_path = os.path.join(self.folder, i)
            tmp_images = np.array([imread(os.path.join(tmp_path, fname)) for fname in os.listdir(tmp_path) if fname.endswith('.jpg')])
            all_images.append(tmp_images)
            all_filenames.append(os.listdir(tmp_path))

        final_array = np.array(all_images)
        return final_array, cat, all_filenames

    # Balancing dataset and dividing into test and train
    # note: this is only used when handcrafting the test size
    def train_test_split(self, image_set, filenames, trainsize = 10, testsize = 10):
        # Sampling random numbers from the size of the image set
        np.random.seed(2022)
        all_elements = np.random.choice(len(image_set), trainsize+testsize, replace=False)

        train_elements = all_elements[:trainsize]
        test_elements = all_elements[-testsize:]

        # Selecting the train elements
        train_images = image_set[train_elements]
        # Selecting filenames for train elements
        train_filenames = [filenames[k] for k in train_elements]

        # Selecting test elements
        test_images = image_set[test_elements]
        # Selecting filenames for test elements
        test_filenames = [filenames[k] for k in test_elements]

        return train_images, test_images, train_filenames, test_filenames

    def sift_descriptors(self, img):
        if len(img.shape) > 2:
                # Grayscale image
            gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            # Sift engine
        sift = cv2.SIFT_create()
        # Key points
        kp = sift.detect(gray, None)
        # Image with keypoints
        # img = cv2.drawKeypoints(gray, kp, test_img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        _ , des = sift.compute(gray, kp)

        return des

    def train_bow(self):

        from sklearn.model_selection import train_test_split
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        # Train test split
        train = []
        train_files = []
        test = []
        test_files = []

        # Applying test_train split to every category
        for image_set, filenames in zip(self.all_images, self.all_filenames):
            train_images, test_images, train_filenames, test_filenames = \
                train_test_split(image_set, filenames, test_size=0.5)
            # self.train_images, train_filenames, test_images, test_filenames = train_test_split(image_set, filenames)
            train.append(train_images)
            train_files.append(train_filenames)
            test.append(test_images)
            test_files.append(test_filenames)

        train = np.array(train)
        train_files = np.array(train_files)

        test = np.array(test)
        test_files = np.array(test_files)

        # The SIFT features should be concatenated into a matrix, one descriptor per row.
        for set_idx, train_sets in enumerate(train):
            for img in train_sets:
                des = self.sift_descriptors(img)
                if set_idx == 0:
                    descriptors = des
                else:
                    #np.append(descriptors, des, axis=0)
                    np.concatenate((descriptors, des), axis=0)
                    #np.vstack((descriptors, des))

        # Then you should run the k-means clustering algorithm on the subset of training descriptors to extract good prototype (visual word) clusters.
        # A reasonable k should be small (say b$etween 200 and 500) for a small number of categories (say 5)
        # and larger (say between 500 and 2000) for a larger number of categories.
        # Also, a good value of k may depend on the complexity of your data.

        # Inspiration from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        # Maybe use this link to demonstrate why scipy is nice https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html#comparison-of-high-performance-implementations
        from scipy.cluster.vq import vq, kmeans, whiten

        # Normalize (by standard deviation)
        whitened = whiten(descriptors)
        self.desc = len(whitened)

        # ensure we don't have more categories than descriptors
        if self.k_val > len(whitened):
            self.k_val = len(whitened)


        # create codebook
        codebook, distortion = kmeans(whitened, self.k_val, seed=2022)

        # Assigns a code from a code book to each observation.
        # Each observation vector in the ‘M’ by ‘N’ obs array is compared with the centroids in the code book and assigned the code of the closest centroid.
        # The code book is usually generated using the k-means algorithm.
        # Each row of the array holds a different code, and the columns are the features of the code.

        # vq returns:
        # code - A length M array holding the code book index for each observation.
        # dist - The distortion (distance) between the observation and its nearest code.
        code_idx, dist = vq(whitened, codebook)

        # Indexing
        # Now construct and save a table that would contain, per entry at least the file name, the true category, if it belongs to the training- or test set, and the corresponding bag of words / word histogram.
        # The table need only be computed once and then used repeatably in the following retrieval experiments.

        # The SIFT features should be concatenated into a matrix, one descriptor per row.

        # initialize train and test lists for bow
        bow_train = []
        bow_test = []

        # Going through all the train sets of each category
        for set_idx, train_sets in enumerate(train):
            # Print the category
            print(self.order[set_idx])

            # Going trhough each image in the category
            for img_idx, img in enumerate(train_sets):
                # Find descriptors
                des = self.sift_descriptors(img)
                # Whiten
                single_whit = whiten(des)
                # Find distribution of codes
                single_code_idx, _ = vq(single_whit, codebook)
                # Histogram
                #c = Counter(single_code_idx)
                imhist = np.zeros((codebook.shape[0]))
                for idx in single_code_idx:
                    imhist[idx] += 1

                # saving a list with the group, the category, the filename and the histogram/bow
                l = ["train", self.order[set_idx], train_files[set_idx][img_idx], imhist]

                bow_train.append(l)

        for set_idx, test_sets in enumerate(test):
            for img_idx, img in enumerate(test_sets):
                # Find descriptors
                des = self.sift_descriptors(img)
                # Whiten
                single_whit = whiten(des)
                # Find distribution of codes
                single_code_idx, _ = vq(single_whit, codebook)
                # Histogram
                imhist = np.zeros((codebook.shape[0]))
                # Histogram approach taken from https://github.com/JaggerWu/images-retrieving/blob/master/CodeBook_generation.py

                for idx in single_code_idx:
                    imhist[idx] += 1
                #c = Counter(single_code_idx)

                l = ["test", self.order[set_idx], test_files[set_idx][img_idx], imhist]

                bow_test.append(l)

        # Now we have a list with histograms of prototypes per train image
        self.bow_train = np.array(bow_train)

        # Now we have a list with histograms of prototypes per test image
        self.bow_test = np.array(bow_test)

    def calculate_dist(self, test_idx):
        euclidean = []
        bhat = []
        x2 = []
        common_words = []
        idf_count = np.zeros(self.bow_train[0][3].shape)
        tf_idf = []

        # Distances
        test_hist = self.bow_test[test_idx]
        tf_test = test_hist[3]/len(test_hist[3])

        for hist in self.bow_train:
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            # kls.append(kl_div(test_hist[2], hist[2]))

            # Euclidean distance
            euclidean.append((-1)*np.sqrt(sum((test_hist[3]-hist[3])**2)))
            # or cv2.norm(test_hist[3], hist[3], normType=cv2.NORM_L2)

            # Bhattacharyya Distance
            bhat.append(sum(np.sqrt(test_hist[3]*hist[3])))
            # bhat.append(np.sqrt(np.linalg.norm(test_hist[3])*np.linalg.norm(hist[3])))

            # Common_words similarity
            test_bool_arr = test_hist[3].astype(bool)
            train_bool_arr = hist[3].astype(bool)
            common_words.append(sum((test_bool_arr == train_bool_arr) & (test_bool_arr == True)))

            # TF-IDF similarity
            tf_train = hist[3] / len(hist[3])
            idf = [2 / (int(test_bool_arr[x])+int(train_bool_arr[x])) if (int(test_bool_arr[x])
                        +int(train_bool_arr[x])) != 0 else 0 for x in range(len(test_bool_arr))]
            tfidf_train = tf_train * idf
            tfidf_test = tf_test * idf
            tf_idf.append(sum((tfidf_train*tfidf_test)))

        return [np.array(euclidean), np.array(bhat), np.array(common_words), np.array(tf_idf)]

    def example_cal(self, test_idx):
        distances = self.calculate_dist(test_idx)

        def max_loc(dist):
            max_loc = np.where(dist == max(dist))
            top = imread(os.path.join("101_ObjectCategories", self.bow_train[max_loc][0][1],
                                      self.bow_train[max_loc][0][2]))
            plt.imshow(top)
            plt.show()

        query_img = imread(os.path.join("101_ObjectCategories", self.bow_test[test_idx][1],
                                        self.bow_test[test_idx][2]))
        plt.imshow(query_img)
        plt.show()

        for dist in distances:
            max_loc(dist)

    # Test all test images and extract mean reciprocal rank etc
    def write_df(self):
        import time

        start = time.time()
        acc = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        match_top3, match, rank = [], [], []

        test_set = self.bow_test
        if self.only_train:
            test_set = self.bow_train

        # iter over dataframe to fill it
        for index, dist_meas in enumerate(acc):

            for idx, test_hist in enumerate(test_set):
                # Distances
                true_cat = test_hist[1]

                # get distance arrays for all different distance measures
                # index 0 == euclidean, 1 == bhat, 2 == common_words, 3 == tf_idf
                dist = self.calculate_dist(idx)

                # take the corresponding distance to the row we're in
                current_dist = dist[index]

                # Find the max value and its position
                max_location = np.where(current_dist == max(current_dist))

                # Take the category of the bow with smallest distance
                predicted_cat = self.bow_train[max_location][0][1]

                # The mean reciprocal rank (i.e. the average across all queries of 1/rank,
                # where rank is the rank position of the first correct category for the i’th query).
                all_pred_cats = [self.bow_train[np.where(current_dist==k)][0][1] for k in sorted(current_dist)]
                all_pred_cats = np.array(all_pred_cats)

                if not np.where(all_pred_cats==true_cat)[0].size == 0:
                    correct_position = np.where(all_pred_cats==true_cat)[0][0]
                else:
                    correct_position = 0

                rank.append(correct_position)

                # Top 3 positions
                top3 = [self.bow_train[np.where(current_dist==k)][0][1] for k in sorted(current_dist)[:3]]

                match.append(true_cat == predicted_cat)
                match_top3.append(true_cat in top3)

            dist_meas[0] = sum(match_top3)/test_set.shape[0]
            dist_meas[1] = sum(match)/test_set.shape[0]
            dist_meas[2] = np.mean(rank)

            # empty temp lists
            match_top3, match, rank = [], [], []

        acc.append([self.desc, self.desc, self.desc])

        end = time.time()
        return acc, (end-start)

    # Main method
    def main(self):
        import pandas as pd
        import datetime

        # create dataframe, save computation time
        data, time = self.write_df()
        test_or_train = ''
        if self.only_train:
            test_or_train = 'train'
        else:
            test_or_train = 'test'
        result = pd.DataFrame(data, index=['euclidean', 'bhat', 'common_words', 'tf_idf', 'features'],
                              columns=['top3', 'match', 'rank'])
        info_str = 'Accuracy for {} train images, {} test images from {} set, {} categories, kmeans with k={}.' \
                   'Computation time: {}'.format(len(self.bow_train), len(self.bow_test), test_or_train, self.categories,
                                                   self.k_val, datetime.timedelta(seconds=round(time, 2)))
        return info_str, result


if __name__=='__main__':
    # i == number of categories
    nlist = [1, 5, 20]
    klist = [200, 500, 2000]

    # try retrieval of only train images
    for i in nlist:
        # k == number of k features
        for k in klist:
            cbir = Cbir("101_ObjectCategories",i, k, True)
            # uncomment following line to see example pictures
            # cbir.example_cal(random.randint(0, len(cbir.bow_test)-1))
            print(cbir.main())

    # do same process for only test images
    for i in nlist:
        # k == number of k features
        for k in klist:
            cbir = Cbir("101_ObjectCategories",i, k, False)
            # uncomment following line to see example pictures
            # cbir.example_cal(random.randint(0, len(cbir.bow_test)-1))
            print(cbir.main())
