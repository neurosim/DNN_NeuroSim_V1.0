# DNN+NeuroSim V1.0

The DNN+NeuroSim framework was developed by [Prof. Shimeng Yu's group](https://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

This is the released version 1.0 (June 1st, 2019) for the tool, and this version is an integrated framework developed in C++ and wrapped by Pytorch and Tensorflow, it supports **_digital SRAM, analog eNVM and FeFET_** based architectures, for deep neural networks (DNNs) inference performance estimation.

In Pytorch/Tensorflow wrapper, users are able to define **_network structures, precision of synaptic weight and neural activation_**. With the integrated NeuroSim which takes real traces from wrapper, the framework can support hierarchical organization from device level to circuit level, to chip level and to algorithm level, enabling **_instruction-accurate evaluation on both accuracy and hardware performance of inference_**.

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:

P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures,*§ IEEE International Electron Devices Meeting (IEDM)*, 2017, San Francisco, USA.

If you have logistic questions or comments on the model, please contact [Prof. Shimeng Yu](mailto:shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact [Xiaochen Peng](mailto:xpeng76@gatech.edu) or [Shanshi Huang](mailto:shuang406@gatech.edu).


## File lists
1. Manual: `Documents/DNNNeuroSim_Manual.pdf`
2. NeuroSim under Pytorch Inference: 'Inference_pytorch/NeuroSIM'
3. NeuroSim under Tensorflow Inference: 'Inference_tensorflow/source/NeuroSIM'


## Installation steps (Linux)
1. Get the tool from GitHub
```
git clone https://github.com/neurosim/DNN_NeuroSim_V1.0.git
```

2. Compile the codes
```
make
```

3. Run Pytorch/Tensorflow wrapper (integrated with NeuroSim)


For the usage of this tool, please refer to the manual.


## References related to this tool 
1. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An Integrated Device-to-Algorithm Framework for Benchmarking Synaptic Devices and Array Architectures,*§ IEEE International Electron Devices Meeting (IEDM), 2017, San Francisco, USA.
2. P.-Y. Chen, X. Peng, S. Yu, ※System-level benchmark of synaptic device characteristics for neuro-inspired computing,§ IEEE SOI-3D-Subthreshold Microelectronics Technology Unified Conference (S3S) 2017, San Francisco, USA.
3. P.-Y. Chen, S. Yu, ※Partition SRAM and RRAM based synaptic arrays for neuro-inspired computing,*§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2016, Montreal, Canada.
4. P.-Y. Chen, B. Lin, I.-T. Wang, T.-H. Hou, J. Ye, S. Vrudhula, J.-S. Seo, Y. Cao, and S. Yu, ※Mitigating effects of non-ideal synaptic device characteristics for on-chip learning,*§ IEEE/ACM International Conference on Computer-Aided Design (ICCAD)*, 2015, Austin, TX, USA.
5. P.-Y. Chen, D. Kadetotad, Z. Xu, A. Mohanty, B. Lin, J. Ye, S. Vrudhula, J.-S. Seo, Y. Cao, S. Yu, ※Technology-design co-optimization of resistive cross-point array for accelerating learning algorithms on chip,*§ IEEE Design, Automation & Test in Europe (DATE)*, 2015, Grenoble, France.
6. N. E. Weste, D. Harris, ※CMOS VLSI Design - A Circuit and Systems Perspective, *§ 2007.
7. X. Peng, R. Liu, S. Yu, ※Optimizing weight mapping and data flow for convolutional neural networks on RRAM based processing-in-memory architecture, *§ IEEE International Symposium on Circuits and Systems (ISCAS), 2019, Sapporo, Japan.
8. W. Khwa, et al., ※A 65nm 4Kb algorithm-dependent computing-in-memory SRAM unit-macro with 2.3ns and 55.8TOPS/W fully parallel product-sum operation for binary DNN edge processors,*§ IEEE International Solid State Circuits Conference (ISSCC), 2018, San Francisco, USA.
9. M. Jerry, et al., ※Ferroelectric FET analog synapse for acceleration of deep neural network training,*§ IEEE International Electron Devices Meeting (IEDM), 2017, San Francisco, USA.
10. X. Sun, et al., ※XNOR-RRAM: A scalable and parallel resistive synaptic architecture for binary neural networks,*§ ACM/IEEE Design, Automation & Test in Europe Conference (DATE), 2018, Dresden.
11. S. Wu, et al. ※Training and inference with integers in deep neural networks,*§ arXiv: 1802.04680, 2018.
12. github.com/boluoweifenda/WAGE
13. github.com/stevenygd/WAGE.pytorch
14. github.com/aaron-xichen/pytorch-playground
