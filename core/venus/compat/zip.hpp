#pragma once

#if defined(_MSC_VER)
// Always use range-v3 zip for MSVC
#include <range/v3/view/zip.hpp>
namespace venus::compat {
using ranges::views::zip;
}
#elif __cplusplus >= 202302L
// C++23: use std::views::zip
#include <ranges>
namespace venus::compat {
using std::views::zip;
}
#else
// C++17/20: use range-v3
#include <range/v3/view/zip.hpp>
namespace venus::compat {
using ranges::views::zip;
}
#endif