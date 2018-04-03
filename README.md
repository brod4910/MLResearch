# Machine Learning Research 5/15/17 - Present

## Backstory

Over the past year, I have had the pleasure of learning and researching machine learning under the supervision of Rajeev Balasubramonian, an excellent professor at the University of Utah. Our goal, alongside another colleague of mine, is to create a power-efficient machine learning chip with little to no accuracy loss. The first half of our research was to learn the ropes of machine learning by reading the Deep Learning Book by Ian Goodfellow. This is a book I would recommend to anyone trying to learn machine/deep learning.

After finishing the book, we started tackling paper after paper. Some of the papers included:  
DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING. https://arxiv.org/pdf/1510.00149.pdf  
Low-Power Cache Design Using 7T SRAM Cell. http://ieeexplore.ieee.org/abstract/document/4155050/  
VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION. https://arxiv.org/pdf/1409.1556.pdf  
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/pdf/1502.03167.pdf  
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. https://arxiv.org/pdf/1602.07261.pdf  

These are only some of the many papers that we tackled. My colleague and I did not start tackling code until around January. I worked on a side project to get myself familiar with LSTM networks but we decided to focus on convolutional neural networks. I made the decision to work with Caffe2 since a lot of the building constructs are somewhat straightforward and powerful. Building a model in Caffe2 is not as difficult the second time around once you understand and have sifted through loads of source code to understand what your model is doing.

Building these models solely to do benchmark testing on different types of CPUs and GPUs. Running these benchmarks tests using the Center for High-Performance Computing (CHPC). The CHPC allows us to submit jobs to train and test these models and measure accuracy. We also use the Nvidia profiler tool to get metrics like power, temperature and time while our application runs. These benchmarks will be compared to our chip's metrics.

## Resources
The dataset used to train these models is the blood-cells dataset from kaggle: https://www.kaggle.com/paultimothymooney/blood-cells  
We decieded to do something a little different from the overkilled MNIST dataset.