import os

path = 'Y:\\DLC\\ACC-DMS_nphr-acb-2020-07-30\\videos'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.mp4' in file:
           files.append(os.path.join(r, file))

for f in files:
   x = f.split('videos\\')[1]
   ssh = 'sbatch job_analyze_videos.sh'
   a = ' "/tigress/acbandi/DLC/ACC-DMS_nphr-acb-2020-07-30/config.yaml"'
   c = ' "/tigress/acbandi/DLC/ACC-DMS_nphr-acb-2020-07-30/videos/'
   print(ssh + a + c + x)