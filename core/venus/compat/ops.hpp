#pragma once
#ifdef VENUS_INTERPRETER
// C++17/20: use range-v3
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
namespace venus::compat {
using ranges::views::transform;
using ranges::views::zip;
} // namespace venus::compat
#else
// C++23: use std::views
#include <ranges>
namespace venus::compat {
using std::views::transform;
using std::views::zip;
} // namespace venus::compat
#endif