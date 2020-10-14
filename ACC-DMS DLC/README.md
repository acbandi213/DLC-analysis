# scripts for analyzing DLC data for ACC-DMS project

# Outlier frame analysis 
percent outliers uncertain_jump.py - finds what percent of overall frames are considered outlier frames using the DLC 'uncertain' and 'jump' outlier extraction method 

job_percent_outliers_uncertain_jump.sh - SSH script for running percent outliers uncertain_jump.py on Tiger Cluster 

percent_outliers_fitting.py - finds what percent of overall frames are considered outlier frames using the DLC 'fitting' outlier extraction method 

job_percent_outliers_fitting.sh - SSh scrpt for running percent_outliers_fitting.py on Tiger Cluster (percent_outliers_fitting.py is computationally intensive and needs to be run on Cluster)

Compare Outlier Frames - Jupityer notebook to make boxplots of percent outlier frame data 

find which frames are outliers for one video.ipynb - Jupityer notebook that takes one video and runs the 3 different extraction algorithm on that one video and outputs lists of the frames identified as outliers from each different algorithm. 

extract outlier frames from video.ipynb - Jupityer notebook that extracts outlier frames identified by each outlier algorithm as well as outlier frames that are common across all 3 outlier algorithms. 

extract_jump_outliers.py - identifies outlier frames using 'jump' algorthm and extracts a certain amount of outlier frames from each video 

extract_outliers_from_all_frames.py - identifies all outlier frames in all videos using jump and then extracts a random amount of outlier frames across all videos 

# Random
find_percent_disenaged_frames.py - sets an ROI and any frame in which the joints are outside the ROI are identified as disengaged 

create ssh names.py - creates the SSH input names to run DLC analyze_videos and make_videos scripts on the Tiger Cluster 
