IPCA
====

This package implements Ideal PCA in MATLAB. 

Ideal PCA is a cross-kernel based feature extraction algorithm which is (a) a faster alternative to kernel PCA and (b) a method to learn data manifold certifying features.

For a derivation of the algorithm and details, see

Franz J. Király, Martin Kreuzer, Louis Theran. *Learning with Cross-Kernels and IPCA*. http://arxiv.org/abs/1406.2646


Getting started
---------------

Copy all files to a directory in your MATLAB path.

The release contains a template script example_script.m which needs to be completed before running. See the file for further explanation.

The IPCA algorithm is encapsulated in two routines:

IPCA pre-computes the cross-kernel matrices and other parameters;

IPCA_eval takes the output of IPCA and converts it into a collection of output features which can be specified by the user, depending on the learning task.

See the inline documentation in IPCA.m and IPCA_eval.m for usage syntax.


License
-------

Copyright (c) 2014, Franz J. Király, Martin Kreuzer, Louis Theran.

This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3, as published by the Free Software Foundation. See the file LICENSE.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.