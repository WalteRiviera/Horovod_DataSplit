# Horovod_DataSplit
One of the current approach used to speed-up training of Neural Networks consists in distributing the computation across multiple resources.
The tool commonly used by the community to make this parallelization happening, is Horovod: https://github.com/horovod/horovod

Horovod is just great to take care of the number of process defined by mpi, distribute the computation and orchestrate the whole process. 
However, the assumption is that the user has already defined a way to split the dataset and is providing the final chunks of data to the respective process.

Before discovering the super-power of Tf.data.Dataset (https://mdw771.github.io/tf-dataset/ ), I used to perform the split manually.
Horovod_DataSplit is a basic example based on publicly available Dataset.

## Running the code
The code is expecting an hdf5 dataset. I used the "Task01_Brain_Tumour.tar" available here:
http://medicaldecathlon.com/

Hope it helps! :)

