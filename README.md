# Introduction

### Description of the problem and a discussion of the background

A relocation is stressful. Ideally, a new hire should be able to fully concentrate on the job requirements without spending lots of time to settle in. A real-estate agency is often approached to find a temporary accommodation for new employees.  Depending on the preferences (profile) of the new hire, the estate agent usually provides the best matches based on the experience (classical approach). However the experience only reflects the past without a projection into the future.

A modern approach based on the data analytics enables to allocate the best possible place for the temporary accommodation based on the following categories: venues,  rental (monthly) costs, crimes, schools (only applies if the new hire has got children). Moreover, the historical data can be used to make some predictions on the neighborhood of the current accommodation,e.g.  w.r.t. crime, to exclude or put a more weight on this particular  neighborhood for the future clients.

The business need: optimize the rent of the temporary accommodation. Better matches imply fully satisfied clients and a quicker turn around. First create a profile of the client based on his current neighborhood. Then search for a best-fit place for the temporary accommodation. Moreover, predictions about  the  neighborhood could save time in decision making, which implies  a better performance.

In this project, I used Berlin as a place to move in.

### Description of the data and how it will be used to solve the problem

Based on the predefined categories (crimes, schools rental costs, venues) collect corresponding data sets (four in total).
Travel sites on the Internet, e.g., FourSquare www.foursquare.com, are ideally  to collect data on city venues.
Governments sites are suitable to collect data about historical crimes. Real estate agency sites are preferable for scrapping the average monthly rent. Local council sites are a good candidate to collect data on the schools, i.e., OFSTED reports.


Venue data were collected data as followed: 
1. Query the FourSqaure website for the top sites in Berlin;
2. Use the FourSquare API to get top restaurants and shopping recommendations closest to each of the top site  within a distance.

Crime data were collected as follows:
1. Use open source crime statistics for Berlin  https://www.statistik-berlin-brandenburg.de/regionalstatistiken/r-gesamt_neu.asp?Ptyp=410&Sageb=12015&creg=BBB&anzwer=6  
2. Download corresponding historical data as an excel file.

Rental costs data were collected as follows:
1. Use https://www.wohnungsboerse.net/mietspiegel-Berlin/2825
2. Scrap for the average monthly rental costs per square meter.

School data were collected as follows:
1. Use https://www.gymnasium-berlin.net/abiturdaten/2018 to collect information on the average grade. Use https://www.gymnasium-berlin.net/adressliste to collect information on the post codes. 
2. Scrap for the average grade of schools in the borough for years 2012-2018.

# Data

#### Data Preparation 

The purpose of this step is a transformation of the collected data into a useable subset.  It composes reading in different excel tables, scrubbing the webpages and saving data as pre-processed  in CSV-files. 

Crime data: the excel table was read in. The first two sheets were ignored. The sheets named  „Fallzahlen\*“  were transformed into panda data frames. The first four rows were dropped in every data frame.  The columns were renamed by translating into english. The data frame was saved to an csv file.

School data: scraping for grades using  BeautifulSoup. Grade data and post codes were merged into a single data frame, which was saved to an csv file.

Rental costs data: scraping for average rental costs using  BeautifulSoup. Rental costs and  post codes were merged into a single data frame, which was saved to an csv file.

Venue data: geographical information consisting post codes for Berlin was collected using https://www.berlinstadtservice.de/xinh/Postleitzahlen_Berlin_Alphabetisch.html Using BeautifulSoup and Requests the results of the Top Pick for Charlottenburg-Wilmersdorf were retrieved.

All temporary data frames were merged into one frame consists of the folowing columns:
Borough, Neighborhood, Longitude, Latitude, School grade, Rental costs, Crime records.


#### Data visualization and descriptive statistics

Descriptive statistics are brief descriptive quantities that summarize a given data set and also allow to provide the first insights on the data. Among others: mean, std, min, max, quantiles. They descriptive statistic could directly be obtained by df.describe(). 

![alt text](https://raw.githubusercontent.com/sergiusdell75/Coursera_Capstone/data_images/img.png)

# Methodology
I use the following machine learning methods: k-nearest neighbors algorithm (k-NN), linear regression, k-means clustering and agglomerative clustering. Below a brief description of used methods is given.

*k-nearest neighbors algorithm or k-NN* (source: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
     1. In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
     2. In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.

k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value (for k-NN regression) is known. This can be thought of as the training set for the algorithm, though no explicit training step is required. A peculiarity of the k-NN algorithm is that it is sensitive to the local structure of the data. 

*Linear regression* (source: https://en.wikipedia.org/wiki/Linear_regression) is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Such models are called linear models. Most commonly, the conditional mean of the response given the values of the explanatory variables (or predictors) is assumed to be an affine function of those values; less commonly, the conditional median or some other quantile is used. Like all forms of regression analysis, linear regression focuses on the conditional probability distribution of the response given the values of the predictors, rather than on the joint probability distribution of all of these variables, which is the domain of multivariate analysis. 

*k-means clustering* (source: https://en.wikipedia.org/wiki/K-means_clustering) is  method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. k-Means minimizes within-cluster variances (squared Euclidean distances), but not regular Euclidean distances, which would be the more difficult Weber problem: the mean optimizes squared errors, whereas only the geometric median minimizes Euclidean distances. Better Euclidean solutions can for example be found using k-medians and k-medoids. 

*Hierarchical cluster analysis or HCA* (source: https://en.wikipedia.org/wiki/Hierarchical_clustering) is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:
    1. Agglomerative: This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
    2. Divisive: This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In general, the merges and splits are determined in a greedy manner. The results of hierarchical clustering are usually presented in a dendrogram.
