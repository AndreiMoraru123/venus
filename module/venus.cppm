module;
#define VENUS_MODULE
#include <cassert>
#ifdef _WIN32
#include <malloc.h>
#endif
export module venus;
// Unexported, Clang misses ADL into std and prints Tensor's ')' as a pointer
export import std;
#ifdef __clang__
// extern "C++" attaches these to the global module, not to venus
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
export extern "C++" {
#include <venus.hpp>
}
