// config_assert.h
#pragma once

#define __assert_fail(expr, file, line, function) throw std::runtime_error("Assertion failed")

// we need this assert in release mode as well...
#define assert(expr)                            \
     (static_cast <bool> (expr)                        \
      ? void (0)                            \
      : __assert_fail (#expr, __ASSERT_FILE, __ASSERT_LINE,             \
                       __ASSERT_FUNCTION))
