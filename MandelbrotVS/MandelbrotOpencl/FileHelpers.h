#pragma once
#include <string>
#include <vector>

std::string LoadTextFile(const std::string& filename);
std::vector<uint8_t> LoadBinaryFile(const std::string& filename);

bool SaveBinaryFile(const std::string folder, const std::string filename, std::vector<uint8_t>& fileContent);