#include "FileHelpers.h"
#include <string>
#include <fstream>
#include <streambuf>
#include <filesystem>

// Performance might not be the best. But still ok for our needs
std::string LoadTextFile(const std::string& filename)
{
  std::ifstream f(filename);
  std::string result((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return result;
}

std::vector<uint8_t> LoadBinaryFile(const std::string& filename)
{
  FILE* f;
  if(fopen_s(&f, filename.c_str(), "rb") == 0)
  {
    std::vector<uint8_t> contents;
    fseek(f, 0, SEEK_END);
    contents.resize(ftell(f));
    rewind(f);
    fread(contents.data(), 1, contents.size(), f);
    fclose(f);
    return contents;
  }
  const auto error = "Failed to load file: " + filename;
  throw std::exception(error.c_str());
}

bool SaveBinaryFile(const std::string folder, const std::string filename, std::vector<uint8_t>& fileContent)
{
  std::error_code errorCode;
  if(!std::filesystem::exists(folder))
  {
    if(!std::filesystem::create_directories(folder, errorCode))
    {
      return false;
    }
  }

  FILE* f;
  if(fopen_s(&f, std::filesystem::path(folder).append(filename).string().c_str(), "wb") == 0)
  {
    const size_t bytesWritten = fwrite(fileContent.data(), sizeof(uint8_t), fileContent.size(), f);
    fclose(f);
    return bytesWritten == fileContent.size();
  }
  return false;
}