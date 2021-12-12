The proposed NN was used for prediction and classification of seven MLST loci (gapA, infB, mdh, pgi, phoE, rpoB, tonB) in squiggles of 29 Klebsiella pneumoniae genomes. 

The proposed NanoGeneNet can be divided into three basic parts: an extraction of features, a gene localization part (so-called a sequence-to-sequence regime) and finally, a gene classification part with another feature extraction, i.e. a sequence-to-vector regime. Using the combination of convolution and recurrent networks turned out to be a great solution for long and unevenly length signals. 

The network was realized in Python 3.9 with the PyTorch library. The source code for training, validation, and especially feedforward of NanoGeneNet is available on GitHub along with a demo and an example squiggle.

The training of NanoGeneNet was performed on computational device with Intel Xeon E5-2603v4, 16 GB RAM and graphical card nVidia Titan Xp, 12 GB GDDR5. The network was realized in Python 3.9 with the PyTorch library. 

That requests the installation of:

