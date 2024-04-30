# Keccak256-Cuda

-Keccak256 hashing algorithm implementation in Cuda and c++.

-Its a standalone version that allows implementation into any project extremly easy.

-main.cpp contain a example usage of computation of the keccak hash of a ethereum public key.

-By removing the first 24 characters of the resulting hash and adding a "0x" u obtain the ethereum address.

-Rng.cu its not necessary !! its just a kernel for generating random 64 characters hex strings (Private keys). if u still need to use it . its just calling it with a int value as argument. it will return the number of keys u asked based on the int value.
