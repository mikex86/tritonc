# tritonc

Your standalone commandline triton compiler.
Write your triton kernels directly in MLIR and compile it to ptx with this handy tool without ever touching python.

## Example:

### add_kernel.ttir

```mlir
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
```

### Commandline
```commandline
tritonc add_kernel.ttir --compute-capability 89 --num-stages 3 --num-warps 4 -o out.ptx
```

### How to build

#### Build LLVM

The required llvm version is a submodule of this repository under `third_party/llvm-project`.
Build-install the llvm-project system-wide.
Make sure to install `ninja-build` for fast builds!

```commandline
cd third_party/llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build .
sudo cmake --build . --target install
```

##### Hot to build tritonc

Make sure you have the CUDA sdk installed and that cuda is in your system include path.
With LLVM built and installed, you can now build tritonc.

```commandline
cd tritonc
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target tritonc -j 14
./tritonc --help
```
