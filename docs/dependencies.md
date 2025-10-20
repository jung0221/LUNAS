# Dependencies

This project depends on the following external components for building and running the native tools:

- CMake (>= 3.15)
- Microsoft Visual Studio 2022 (or newer) with C++ workload for Windows builds
- zlib (used by NIfTI code paths)
- (Optional) OpenMP for faster builds and parallelism

Recommended local install locations (Windows examples used in this documentation):

- zlib: `C:\libs\zlib` (includes `include\zlib.h` and `lib\zs.lib` or `lib\z.lib` plus `bin\z.dll`)

Why place zlib under C:\libs

- Using a fixed, simple path reduces CMake discovery issues and lets the provided CMake helper prefer a static library when available.

If you prefer to use system-installed zlib or package manager installations, adapt the CMake command-line flags accordingly using `-DZLIB_ROOT` / `-DZLIB_INCLUDE_DIR` / `-DZLIB_LIBRARY`.