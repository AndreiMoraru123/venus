#pragma once
#ifdef VENUS_INTERPRETER
// MSVC: use std::views
#include <ranges>
namespace venus::compat {
using std::views::transform;
using std::views::zip;
} // namespace venus::compat
#else
// C++23: use std::views
#include <ranges>
namespace venus::compat {
using std::views::transform;
using std::views::zip;
} // namespace venus::compat
#endif