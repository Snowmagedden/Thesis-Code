#!/usr/bin/env python
# coding: utf-8


import os
import os.path


#First make a text document that has all train videos and a document with all test videos

# @Harvey
train_file = 'train.txt'
test_file = 'test.txt'
    
# Build the train list. Extra step to remove the class index.
with open(train_file) as fin:
    train_list = [row.strip() for row in list(fin)]
    train_list = [row.split(' ')[0] for row in train_list]

# Build the test list.
with open(test_file) as fin:
    test_list = [row.strip() for row in list(fin)]

# Set the groups in a dictionary.
file_groups = {
    'train': train_list,
    'test': test_list
}

for group, videos in file_groups.items():
    for video in videos:
        parts = video.split(",")
        classname = parts[1]
        filename = parts[0]


# @Harvey
def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    for group, videos in file_groups.items():

        # Do each of our videos.
        for video in videos:

            # Get the parts.
            parts = video.split(",")
            classname = parts[1]
            filename = parts[0]

            # Check if this class exists.
            if not os.path.exists(os.path.join(group, classname)):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join(group, classname))

            # Check if we have already moved this file, or at least that it
            # exists to move.
            if not os.path.exists(filename):
                print("Can't find %s to move. Skipping." % (filename))
                continue

            # Move it.
            dest = os.path.join(group, classname, filename)
            print("Moving %s to %s" % (filename, dest))
            os.rename(filename, dest)

    print("Done.")


move_files(file_groups)

