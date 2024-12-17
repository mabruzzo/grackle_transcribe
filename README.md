This is a quick and dirty tool to aid with transcribing the Fortran files of [Grackle](https://github.com/grackle-project/grackle) from Fortran to a minimal subset of C++.

## Overview

In case you are randomly stumbling onto this project:

- this is **NOT** a generalized parser translator of Fortran (it is tuned to the dialect used by Grackle). It may also not fully work, but the goal is to help get us most of the way to a transcription (and conservatively identify sections of code that can't be translated)
- currently, we are trying not to target any advanced C++ features. Currently, we are trying to use a minimal feature-set that we find useful. After the transcription, we may shift to a more C-like subset of Grackle


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

This tool has been designed to work with a particular branch of Grackle that can be found [here](https://github.com/mabruzzo/grackle/tree/gen2024-transcribe). This branch is called `gen2024-transcribe` (some additional details are provided at the top of its README).

The most important change made in this branch is that a custom ``MASK_TYPE`` type has replaced nearly every occurence of ``logical``.

## Running the tests

To run the (limited) tests you need to have pytest installed.

You can run it from the root of the repository with:

```
pytest --grackle-src-dir=/path/to/grackle/src/clib
```

## How it works

This tool assumes that some tweaks have been applied to Grackle in order to ease the burden of transcription.

### Getting Subroutine Declarations

You need to generate a C-header that declare C function-signatures for all fortran subroutines. From the root of this repository, invoke:

```sh
python transcribe.py declarations --grackle-src-dir=/path/to/grackle/src/clib
```

An example of the transcribed file can be found [here](https://pastebin.com/qFf2bKK2).

### Quick Transcription Example

Then you can use the ``transcribe.py`` script to generate a source and header file for the first function in a given file.

Here we provide 2 quick examples of invocations (jump to the next section for more sophisticated examples -- we also show some examples of the results).

```sh
python transcribe.py convert \
    --grackle-src-file=/path/to/grackle/src/clib/solve_rate_cool_g.F \
    --use-C-linkage
```

OR

```sh
python transcribe.py convert \
    --grackle-src-file=/path/to/grackle/src/clib/cool_multi_time_g.F \
    --use-C-linkage
```

> [!NOTE]  
> I have confirmed that both invocations produce C++ code that succesfully compiles with C++ 17. I have also confirmed that using the second set of C++ code passes all of the tests[^5].

### Transcription (argument reduction)

It is common for Grackle's highest level Fortran subroutines to take extremely long argument-lists (up to 469 args), where most arguments are extracted from members of Grackle's primary structs (i.e. ``grackle_field_data``, ``chemistry_data``, ``chemistry_data_storage``, ``chemistry_data_storage``, ``code_units``, ``photo_rate_storage``).

As such, we have built functionality into ``transcribe.py`` to reduce the number of arguments being passed into the subroutines. The idea is simple:
- we inform ``transcribe.py`` of a C/C++ code snippet that calls the subroutine
  being translated with the ``--fn-call-loc``
- we also need to tell the tool about any local variables that are defined in the same scope as that C/C++ code snippet and refer to the standard grackle data-types (if we forget to do this the ``transcribe.py`` can't reduce arguments

With this information, ``transcribe.py``:
- essentially, ``transcribe.py`` will eliminate any argument that is directly read from a struct.
- require pointers to these structs to be passed as new arguments
- appropriately replace any reference within the function to the eliminated arguments with accesses to the values directly from the structs.

Here are 2 examples 

```sh
python transcribe.py convert \
    --grackle-src-file=path/to/src/clib/solve_rate_cool_g.F \
    --use-C-linkage \
    --fn-call-loc=path/to/src/clib/solve_chemistry.c:246:716 \
    --grackle_field_data_ptr=my_fields \
    --chemistry_data_ptr=my_chemistry \
    --chemistry_data_storage_ptr=my_rates \
    --code_units=my_units \
    --photo_rate_storage_ptr '&my_uvb_rates'
```

Here's are links to the resulting [source file](https://pastebin.com/bATxZL8D) and [header file](https://pastebin.com/e5WEgeKJ). For reference, [this](https://github.com/mabruzzo/grackle/blob/gen2024-dev/src/clib/solve_rate_cool_g.F) is a link to the original Fortran code that was transcribed.

OR

```sh
python transcribe.py convert \
    --grackle-src-file=path/to/src/clib/cool_multi_time_g.F \
    --use-C-linkage \
    --fn-call-loc=path/to/src/clib/calculate_cooling_time.c:213:542 \
    --grackle_field_data_ptr=my_fields \
    --chemistry_data_ptr=my_chemistry \
    --chemistry_data_storage_ptr=my_rates \
    --code_units=my_units \
    --photo_rate_storage_ptr '&my_uvb_rates'
```

Here's are links to the resulting [source file](https://pastebin.com/0y2nekQN) and [header file](https://pastebin.com/myEYZQJi). For reference, [this](https://github.com/mabruzzo/grackle/blob/gen2024-dev/src/clib/cool_multi_time_g.F) is a link to the original Fortran code that was transcribed.

The former reduces the number of arguments from 469 to 11 and the latter call reduces the arguments from 325 down to 10. I have confirmed that both invocations produce C++ code that succesfully compiles with C++ 17. Again, I have also confirmed that using the second set of C++ code passes all of the tests.


> [!NOTE]  
> In a sense, blind usage of the argument-reduction-functionality exchanges the "code smell" of too many argumnents for the "code smell" of passing more information than is necessary to a function. This is probably a worthwhile tradeoff (especially for functions close to the top of the call stack) since the former "code smell" makes refactoring a very tedious/error-prone process.[^6]

The ``--use-C-linkage`` argument is used to add annotations to control add ``extern "C"`` annotations to the transcribed code to ensure that the transcribed function can be called from C code
- the block controls calling conventions/name mangling
- This is necessary for these particular examples since the transcribed functions are currently called from ``solve_chemistry.c`` and ``calculate_cooling_time.c`` (which are currently C files).
- If a transcribed subroutine doesn't need to be called from C, (i.e. it is onlycalled from transcribed C++ functions) then we can skip this argument.

### Actually using the Transcribed Code

These generated files assume that the following files are also included in the build-process (you need to copy them into source-directory)
- the [utils-cpp.C](other_files/utils-cpp.C) and [utils-cpp.hpp](other_files/utils-cpp.hpp) (from this repository's ``other_files`` subdirectory). These implement some useful C++ functionality to simplify transcription (we could/should trim this down to some degree in the future).
- the generated previously mentioned header holding c-declarations of every subroutine. (we decribed how to generate that file [here](#getting-subroutine-declarations))

You also need to ensure that the C++ files are compiled with the ``--std=c++17`` option (you will probably get errors if you don't do this).

*TODO: add more details about how to modify build-system and how to modify the source files*

## General Philosophy

The basic philosophy is to perform these translations one subroutine at a time, from the top of the call-stack down to the bottom.

We also want to use previously completed work: 
- like the proposed transcriptions of interpolators from the grackle-project/grackle#160 PR
- or maybe the some of the transcriptions of in the grackle-project/grackle#153 PR (this may be more tricky to include)

[^1]: This code was originally written just to be used myself. I hadn't really intended to share it with anybody else. Additionally, development involved a lot of experimentation. Furthermore, the scope of the project creeped a lot (the original intention was to write some tools to help with transcribing chunks of code).

[^2]: Corners were cut in terms of parsing and translation. Some corners were also cut with creating temporary files.

[^3]: In particular, the main execution script should be better packaged within the project.

[^4]: We use ``flang-new`` to dump it's abstract syntax tree (AST). Currently, we use the AST for digesting the declaration section of each subroutine. Originally, I planned to exclusively use the AST for everything. But, I quickly realized that we wanted to retain some meta information that gets translated by the preprocess (such as the ``R_PREC`` type or the names of macros encoding clusters) as well as comments. For that recent, I wrote a simple tokenizer, with the intention of matching each line of code with the AST. But as the scope grew, eventually I ended up with a fairly feature-complete custom parser (for Grackle's Fortran dialect). At this point, we could probably rewrite the declaration-parsing code and remove all dependence on ``flang-new``.

[^5]: Be advised, at the time of testing some of the tests were a little flaky. But if I ran the tests twice, everything would pass. This is an issue the gen2024 branch (even without transcribed code). It is high on my TODO-list to sit down with Valgrind and resolve this issue.

[^6]: In the long term, we should consider introducing more structs. For example we may want to use additional structs (or reorganize existing structs) inside of ``chemistry_data_storage`` so that we can pass chunks of information around (rather than the full struct). Additionally, we may also want to consider copying data specified within the public ``chemistry_data`` struct (we need to be careful about modifying its internals so we don't break API) to package the information into smaller structs (so that we don't also need to pass that whole thing around).
