# Generalizing Convolution to Point Clouds

**Authors**: Davide Bacciu, Francesco Landolfi

This repository contains the code used to obtain the results presented in the [article](https://doi.org/10.14428/esann/2024.ES2024-145).

## Abstract

Convolution, a fundamental operation in deep learning for structured grid data like images, cannot be directly applied to point clouds due to their irregular and unordered nature. Many approaches in literature that perform convolution on point clouds achieve this by designing a convolutional operator from scratch, often with little resemblance to the one used on images. We present two point cloud convolutions that naturally follow from the convolution in its standard definition popular with images. We do so by relaxing the indexing of the kernel weights with a "soft" dictionary that resembles the attention mechanism of the transformers. Finally, experimental results demonstrate the effectiveness of the proposed relaxations on two benchmark point cloud classification tasks. 

## Reference

```bibtex
@inproceedings{bacciu_generalizing_2024,
	location = {Bruges (Belgium) and online},
	title = {Generalizing Convolution to Point Clouds},
	isbn = {978-2-87587-090-2},
	url = {https://www.esann.org/sites/default/files/proceedings/2024/ES2024-145.pdf},
	doi = {10.14428/esann/2024.ES2024-145},
	eventtitle = {{ESANN} 2024},
	pages = {23--28},
	booktitle = {{ESANN} 2024 proceesdings},
	publisher = {Ciaco - i6doc.com},
	author = {Bacciu, Davide and Landolfi, Francesco},
	urldate = {2025-08-31},
	date = {2024},
	langid = {english},
}
```
