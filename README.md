This is a quick and dirty tool to aid with transcribing the Fortran files of [Grackle](https://github.com/grackle-project/grackle) from Fortran to a minimal subset of C++.

## Overview

In case you are randomly stumbling onto this project:

    * this is **NOT** a generalized parser translator of Fortran (it is tuned to the dialect used by Grackle). It may also not fully work, but the goal is to help get us most of the way to a transcription (and conservatively identify sections of code that can't be translated)

    * currently, we are trying not to target any advanced C++ features. Currently, we are trying to use a minimal feature-set that we find useful. After the transcription, we may shift to a more C-like subset of Grackle


> [!IMPORTANT]  
> Due to the very unorthodox development process,[^1] the code in this repository is pretty messy, a lot of corners were cut[^2], and the packaging is somewhat undesirable.[^3]

# Installation

## Dependencies

A minimum python version of 3.10 is needed. Other python dependencies will be automatically installed as part of the installation process.

Currently, this tool requires the use of ["LLVM Flang"](https://flang.llvm.org/docs/) (aka "new Flang"). **Do not confuse this with "Classic Flang"**. To install it on macOS, I invoked

```sh
brew install flang
```

All work has been tested with version 19.1.3. In the future, we could probably remove this dependency.[^4]

## Installing the Package

To run this code, you need to clone this repository. Then you should install the module, from the root of the project directory, with a command like the following:

```sh
pip install -e .
```


# Using the tool

This tool has been designed to work with a particular branch of Grackle.

> [!NOTE]  
> TODO: describe that branch

## Running the tests

To run the (limited) tests you need to have pytest installed.

You can run it from the root of the repository with:

```
pytest --grackle-src-dir=/path/to/grackle/src/clib
```

## How it works

You need to generate a header with C-declarations for all fortran headers. From the root of this repository, invoke:

```sh
python declare_fortran_signatures.py --grackle-src-dir=/path/to/grackle/src/clib
```

Then you can use the ``transcribe.py`` script to generate a source and header file for the first function in a given file. An invocation might look like:

```sh
python transcribe.py --grackle-src-dir=/path/to/grackle/src/clib --fname=solve_rate_cool_g.F
```

OR

```sh
python transcribe.py --grackle-src-dir=/path/to/grackle/src/clib --fname=cool_multi_time_g.F
```

> [!NOTE]  
> I have confirmed that both invocations produce C++ code that succesfully compiles with C++ 17. However, I have not checked correctness, yet.

These generated files assume that the the source files ``other_files/utils-cpp.C`` and ``utils-cpp.hpp`` will also be included in the same build process.


[^1]: This code was originally written just to be used myself. I hadn't really intended to share it with anybody else. Additionally, development involved a lot of experimentation. Furthermore, the scope of the project creeped a lot (the original intention was to write some tools to help with transcribing chunks of code).

[^2]: Corners were cut in terms of parsing and translation. Some corners were also cut with creating temporary files.

[^3]: In particular, the main execution script should be better packaged within the project.

[^4]: We use ``flang-new`` to dump it's abstract syntax tree (AST). Currently, we use the AST for digesting the declaration section of each subroutine. Originally, I planned to exclusively use the AST for everything. But, I quickly realized that we wanted to retain some meta information that gets translated by the preprocess (such as the ``R_PREC`` type or the names of macros encoding clusters) as well as comments. For that recent, I wrote a simple tokenizer, with the intention of matching each line of code with the AST. But as the scope grew, eventually I ended up with a fairly feature-complete custom parser (for Grackle's Fortran dialect). At this point, we could probably rewrite the declaration-parsing code and remove all dependence on ``flang-new``.
