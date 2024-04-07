#define CATCH_CONFIG_MAIN
#include "../catch2/catch.hpp"

#include <random>

#include "test.hpp"

std::mt19937_64 gen(time(nullptr));
