#pragma once
#include "../src/utils.h"
#include <vector>

bool elementwise(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr, std::vector<int> in_shape0,
        std::vector<int> in_shape1, std::vector<int> out_shape, std::string dtype, std::string layout);

