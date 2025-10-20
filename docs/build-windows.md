# Building on Windows with CMake and MSVC

This guide explains how to build the native `gft` library and the `oiftrelax` executable on Windows using CMake + Microsoft Visual C++ (MSVC). It shows the steps I used during porting and the common pitfalls.

Prerequisites

- Visual Studio 2022 with the "Desktop development with C++" workload installed
- CMake 3.15+
- Git
- zlib source to build and install locally (recommended target: `C:\libs\zlib`)

Quick summary of steps

1. Install zlib and place it under `C:\libs\zlib` (see the Dependencies doc).
2. Configure the project with CMake providing the zlib paths.
3. Build the `oiftrelax` target with CMake/Visual Studio.

Detailed steps

1) Build and install zlib (recommended)

- Clone zlib and configure with CMake. Example commands (run in an elevated Developer Command Prompt):

```cmd
mkdir C:\libs
git clone https://github.com/madler/zlib.git C:\libs\zlib-src
cmake -S C:\libs\zlib-src -B C:\libs\zlib-build -DCMAKE_INSTALL_PREFIX=C:\libs\zlib -G "Visual Studio 17 2022" -A x64
cmake --build C:\libs\zlib-build --config Release --target INSTALL
```

After the install you should have `C:\libs\zlib\include\zlib.h` and one or more libs under `C:\libs\zlib\lib` (for example `zs.lib`, `z.lib`) and DLLs under `C:\libs\zlib\bin`.

2) Configure the ctSegmentation project

From the repository root run:

```cmd
cmake -S . -B build -DZLIB_ROOT=C:/libs/zlib -DZLIB_INCLUDE_DIR=C:/libs/zlib/include -DZLIB_LIBRARY=C:/libs/zlib/lib/zs.lib -DCT_USE_STATIC_ZLIB=ON
```

Notes:
- We prefer a static zlib (`zs.lib`) when available; if not present the CMake configuration falls back to a normal ZLIB target.
- If you installed zlib elsewhere adapt `-DZLIB_ROOT` and other flags accordingly.

3) Build the executable

```cmd
cmake --build build --config Release --target oiftrelax
```

If everything configures and builds successfully you'll find `oiftrelax.exe` in `build\src\Release`.

4) Running the program

- If you linked statically to zlib (preferred) there is no runtime zlib DLL to copy. If you linked dynamically you need `z.dll` or `zlib1.dll` on PATH or in the same folder as the executable.

Alternative: Copy zlib DLL(s) to the exe folder after build

```cmd
copy C:\libs\zlib\bin\z.dll build\src\Release\
```

Troubleshooting

- "zlib.h: No such file or directory" — ensure you passed `-DZLIB_INCLUDE_DIR` or that CMake found ZLIB_ROOT.
- Missing `z.dll` at runtime — either add `C:\libs\zlib\bin` to PATH or copy the DLL to the executable folder, or prefer the static libs.
- If you see many compiler warnings from `gft` sources they are mostly benign porting warnings (signed/unsigned, narrowing). Focus on errors (not warnings) first.

If you want I can add a CMake post-build step to copy zlib DLLs into the exe folder automatically when static linking is not selected.