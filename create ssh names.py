#script that will generate the ssh commands for tracking DLC behavior videos on Tiger cluster 
import os

path = 'Y:\\DLC\\ACC-DMS_nphr-acb-2020-07-30\\videos' #input video file path - include \\
ssh = 'sbatch job_analyze_videos.sh' #interchange with job_make_videos.sh
config = ' "/tigress/acbandi/DLC/ACC-DMS_nphr-acb-2020-07-30/config.yaml"' #input config file path 
vid_path = ' "/tigress/acbandi/DLC/ACC-DMS_nphr-acb-2020-07-30/videos/' #input the vid file path without \\ - Tiger reads only / 

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.mp4' in file:
           files.append(os.path.join(r, file))

for f in files:
   vid = f.split('videos\\')[1]
   print(ssh + config + vid_path + vid)
