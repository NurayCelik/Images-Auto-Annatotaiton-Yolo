from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


video_file = "D61_1_1.mp4"

with open("times.txt") as f:
  times = f.readlines()

times = [x.strip() for x in times] 

for time in times:
  sttime = int(time.split("-")[0])
  entime = int(time.split("-")[1])
  ffmpeg_extract_subclip(video_file, sttime, entime, targetname=str(times.index(time)+1)+".mp4")
 
