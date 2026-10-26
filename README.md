# brewpy

Python bindings for the BREW C++ tracking library. The binding generator and
handwritten bindings preserve the local implementation, including unpublished
working-tree updates. They are generated from the same `@mex`-annotated BREW
headers as the MATLAB bindings.

## Dependency layout

```text
brewpy/
  brew/                    BREW Git submodule (add before the first commit)
  brewpy/__init__.py       Python package
  generator_files/        Binding generator and handwritten bindings
  CMakeLists.txt
  requirements.txt
```

The prepared folder intentionally has no `brew/` directory or Git metadata.
Add the published BREW repository as the `brew` submodule before committing it.
After cloning the completed repository, initialize nested dependencies:

```sh
git submodule update --init --recursive
```

## Build

Use Python 3.10+, a C++20 compiler, CMake 3.20+, and installed Eigen 3.4 headers.
In a virtual environment, from this checkout:

```sh
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build build --parallel 2
python -c "import brewpy; print(brewpy.models.TemplatePose)"
```

PowerShell configuration uses the same CMake arguments on one line, or backticks
for line continuation. CMake copies the compiled extension into `brewpy/`.
To import it from elsewhere, include this repository root on `PYTHONPATH`.
Build the extension for the Python version and operating system that will run it;
compiled extensions are ignored by Git.

The application builds require installed Eigen and pybind11 and do not fetch
source code. BREW's tests, examples, and C++ plotting are disabled for the Python
build. Their source remains in the standalone BREW dependency. Both applications
use the same dependency revisions; application-specific scripts live outside
these packages.
