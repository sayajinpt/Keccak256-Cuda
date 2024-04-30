#include <iostream>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <string> 
#include <cuda_runtime.h>
#include "keccak256.cu"

int main() {
    //public key
    const std::string publicKey = "3855bbe6b83ad879b300974446094f9b6ec354951799efa21cb59f424dd6546dc464c0f6d689029a07409f30a9a1fa691d591f63718c5e94e59cb68374f8125f";

    // Convertion hexadecimal string to bytes
    std::vector<uint8_t> publicKeyBytes;
    for (size_t i = 0; i < publicKey.size(); i += 2) {
        std::string byteString = publicKey.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byteString, nullptr, 16));
        publicKeyBytes.push_back(byte);
    }

    //Calculate the Keccak256 hash
    uint8_t hashResult[Keccak256::HASH_LEN];
    Keccak256::getHash(publicKeyBytes.data(), publicKeyBytes.size(), hashResult);

    //Print the hash result
    std::cout << "Keccak256 hash: ";
    for (int i = 0; i < Keccak256::HASH_LEN; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hashResult[i]);
    }
    std::cout << std::endl;

    return 0;
}
