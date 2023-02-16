import os
import shutil

def CheckFolder(root, EpName, FolderName, NewOne=True):
    # if exist then remove
    if NewOne:
        if os.path.exists(os.path.join(root, EpName, FolderName)):
            shutil.rmtree(os.path.join(root, EpName, FolderName))
        # make new one
        os.makedirs(os.path.join(root, EpName, FolderName))

    if not os.path.exists(os.path.join(root, EpName, FolderName)):
        # make new one
        os.makedirs(os.path.join(root, EpName, FolderName))
