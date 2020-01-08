import numpy as np
import h5py
import os
import random

hdf5_filename = "../../Task01_BrainTumour.h5"      # Dataset is publicly available: please check Readme

def get_range(mpi_np=2,num_imgs=0):
   # Goal: return the num. of images required to have an even distribution across the mpi_np procs.
 
   if (num_imgs == 0):
      return -1

   else:
      # Before splitting, we need to make sure that the num_img is evenly divided by mpi_np.
      # If not, we'll drop m images in order to get an even dist.
      to_drop = num_imgs % mpi_np

   valid_range = (num_imgs - to_drop)/ mpi_np

   return int(valid_range)


def get_indexes(mpi_np=2, valid_range=0):

   # Goal: generate "mpi_np" lists of unique indexes so that we can then load
   # - For each mpi_np process - an even subset of images from dataset to use to feed hvd

   if (valid_range == 0):
      return -1

   else:
      avail_indexes = set(range(1,valid_range*mpi_np))
      INDEXES = {}
      for p in range(0, mpi_np-1):
         INDEXES[p]=random.sample(avail_indexes,valid_range)
         avail_indexes = avail_indexes-set(INDEXES[p])

      INDEXES[mpi_np-1]=list(avail_indexes)

   return INDEXES


########### FILE ACCESS #############
DIM={}
with h5py.File(hdf5_filename, 'r') as f:
   for k in f.keys():
      print(k, " dimensions:   {}".format(f["imgs_train"].shape))
      DIM[k] = f[k].shape[0]





def main():
   """Easy test to run"""

   mpi_np = 4 		# Setting an example of 4 mpi processes

   imgs = int(DIM["imgs_train"] )	
   imgXproc = get_range(mpi_np, imgs)

   print("Expected ",imgXproc, " img per proc.")
   IDX = get_indexes(mpi_np, imgXproc)

   for i in IDX:
      print("Unique img indexes for HVD proc.", i, " = ", len(IDX[i]))

if __name__ == "__main__":
   main()

