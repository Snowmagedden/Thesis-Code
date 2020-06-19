#!/usr/bin/env python
# coding: utf-8


import csv
import glob
import os
import os.path
from subprocess import call
import cv2


# @Harvey
def extract_files():
    data_file = []
    folders = ['train', 'test']

    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))

        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.avi'))

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path) 

                train_or_test, classname, filename_no_ext, filename = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
#                 if not check_already_extracted(video_parts):
#                     # Now extract it.
                count = 0
                cap = cv2.VideoCapture(video_path) #video_path works wrong folder though
                i=0
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if i%25 == 0:      # total of 3000 frames per image   (i = 3000 %25 is divided by 25 so this gives 120 frames)
                        filename_image = filename + "_frame%d.jpg" % count;count+=1
                        cv2.imwrite(os.path.join(train_or_test,classname, filename_image),frame)
                    i+=1
                cap.release()
                cv2.destroyAllWindows()   
                            
                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(train_or_test, classname,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0] #gets rid of .avi
    classname = parts[1]
    train_or_test = parts[0]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:
    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()