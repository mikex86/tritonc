#pragma once

#include <string>
#include <unordered_map>

std::string applyTemplate(const std::string &codeTemplate,
                          const std::unordered_map<std::string, int> &autotuneValues);