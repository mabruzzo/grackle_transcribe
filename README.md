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

# Recommended Control Flow For transcribing a routine

This tool has been designed to work with a particular branch of Grackle that can be found [here](https://github.com/mabruzzo/grackle/tree/gen2024-transcribe).

Suppose that we want to translate `cool1d_multi_g`

## Running the tool

You should invoke someth like the following. In the following, you will need to replace `${GRACKLE_DIR}` with the path to the Grackle clib

```
python transcribe.py \
   convert \
   --grackle-src-file=${GRACKLE_DIR}/cool1d_multi_g.F \
   --no-use-C-linkage \
   --fn-call-loc=${GRACKLE_DIR}/fortran_func_wrappers.hpp:84:188 \
   --preparse-signatures \
   --var-ptr-pairs 'my_fields,grackle_field_data_ptr' \
       'my_chemistry,chemistry_data_ptr' \
       'my_rates,chemistry_data_storage_ptr' \
       '&idx_range,IndexRange_ptr' \
       '&my_uvb_rates,photo_rate_storage_ptr' \
       '&internalu,InternalGrUnits_ptr' \
       '&grain_temperatures,GrainSpeciesCollection_ptr' \
       '&logTlininterp_buf,LogTLinInterpScratchBuf_ptr' \
       '&cool1dmulti_buf,Cool1DMultiScratchBuf_ptr' \
       '&coolingheating_buf,CoolHeatScratchBuf_ptr' \
       '&species_tmpdens,SpeciesCollection_ptr' \
       '&kcr_buf,ColRecRxnRateCollection_ptr' \
       '&kshield_buf,PhotoRxnRateCollection_ptr' \
       '&chemheatrates_buf,ChemHeatingRates_ptr' \
       '&grain_growth_rates,GrainSpeciesCollection_ptr'
```

This should create a file called `cool1d_multi_g-cpp.C` and a file called `cool1d_multi_g-cpp.h` (really, it should be `cool1d_multi_g-cpp.hpp` since we don't bother giving this C-linkage).

Some added context:
- To help provide the tool with a hint for how to transcribe the routine, we told it about a place (in ``fortran_func_wrappers.hpp``) where the fortran-version is currently called. Ideally, a given routine should only be tranlated after all of the ``FORTRAN_NAME(<routine_name>)(args...)`` calls have been consolidated behind a single C++ wrapper function.
- the ``--var-ptr-pairs`` arguments tells the tool about some structs that arguments were extracted from. The tool is smart enough to make those types arguments in the newly generated C++ code (the decision whether to pass structs by pointer or value is handled internally -- we aren't super consistent about doing this. The original fortran code was roughly equivalent to passing all structs by value)
- in this case, we passed a bunch of names into ``--var-ptr-pairs`` that aren't actually used. That's OK!

When the script is run, information about what the new routine's signature looks like and how it should be called (at the specified location in ``fortran_func_wrappers.hpp``). (Some garbage debugging data might also be printed -- sorry about that!)

## First Steps with the output

1. The first thing you should do is go through the generated source file and find all lines starting with ``//_// PORT:``. These are lines that the tool can't currently translate. You should manually translate these lines yourself (or you add support into the tool for doing this -- if so, please open a PR)

2. You should move the new files into the grackle repository, add the new source-file to CMakeLists.txt and `src/clib/Make.config.objects`. And make sure things compile. (the `-cpp.C` and `-cpp.hpp` suffixes may seem redundant, but don't change that unless you're explicitly willing to check that the new versions don't break the classic build-system). Now might be a good time to commit.

3. Now we need to modify Grackle to actually call the new routine. I would recommend:
   - removing the old Fortran file from the repository
   - modifying CMakeLists.txt and `src/clib/Make.config.objects`
   - and modifying the C++ wrapper so that it forwards onto the newly transcribed version of the function
   - If the code compiles, I would definitely make a commit

4. Before you do anything else, you should confirm that the integration tests all pass. The tests all use [Grackle's answer testing framework](https://grackle.readthedocs.io/en/latest/Testing.html)

   - If you haven't ever run the tests before you are going to need to checkout a known working older version (probably from the commit just before the ones that you made) and follow these instructions [here](https://grackle.readthedocs.io/en/latest/Testing.html#tests-with-answer-verification). In short, after you checkout the older version, you should be able to run the following from the root of the grackle repository (you may want to do this in a virtual environment):

     ```
     $ pip uninstall pygrackle

     # the following touch command shouldn't be necessary, but I think it may
     # help avoid an issue that I once had where pip thinks it can skip
     # some steps in an editable install
     $ touch pyproject.toml

     # the -e flag is important
     $ pip install -e .

     $ GENERATE_PYGRACKLE_TEST_RESULTS=1 pytest src/python/tests/
     ```

     - The ``test_code_examples`` tests will all be skipped if you followed the above (this is currently unavoidable when you build pygrackle in this particular way and on my list of things to fix).

     - The ``test_model[yt_grackle-0-0]`` test will appear to fail. This is because it expects you to download an example dataset and export the ``YT_DATA_DIR`` environment variable to specify its location. It's on my todo list to make us gracefully skip this test (if you really want to run it, look at the logic in ``.circleci``.

     - All other tests should "pass" (for the answer tests, this just means we succesfully saved the answers)

   - Once you have saved the answers, you should return to the latest commit and run the answer tests again. You should probably follow the documentation.
     I have reproduced some quick instructions to run the tests down below (if you follow these instructions, the same tests will be skipped and run into issues).

     ```
     $ pip uninstall pygrackle

     # the following touch command shouldn't be necessary, but I think it may
     # help avoid an issue that I once had where pip thinks it can skip
     # some steps in an editable install
     $ touch pyproject.toml

     # the -e flag is important
     $ pip install -e .

     # if you preciesly followed the previous snippet, the following shouldn't
     # be necessary, but better to be safe than sorry. If the following variable
     # has a value of 1, then you will be overwriting the test answers
     unset GENERATE_PYGRACKLE_TEST_RESULTS

     $ pytest src/python/tests/
     ```

At this point, if all tests pass, you are ready to start cleaning the code.

Some general advice:
- I would err on the side making many atomic commits and smaller well organized PRs


# (OLD) Using the tool

> [!IMPORTANT]  
> This is a little out of date now that we have gotten underway.


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
    --var-ptr-pairs 'my_fields,grackle_field_data_ptr' \
        'my_chemistry,chemistry_data_ptr' \
        'my_rates,chemistry_data_storage_ptr' \
        '&my_uvb_rates,photo_rate_storage_ptr' \
        'my_units,code_units_ptr'
```

Here's are links to the resulting [source file](https://pastebin.com/bATxZL8D) and [header file](https://pastebin.com/e5WEgeKJ). For reference, [this](https://github.com/mabruzzo/grackle/blob/gen2024-dev/src/clib/solve_rate_cool_g.F) is a link to the original Fortran code that was transcribed.

OR

```sh
python transcribe.py convert \
    --grackle-src-file=path/to/src/clib/cool_multi_time_g.F \
    --use-C-linkage \
    --fn-call-loc=path/to/src/clib/calculate_cooling_time.c:213:542 \
    --var-ptr-pairs 'my_fields,grackle_field_data_ptr' \
        'my_chemistry,chemistry_data_ptr' \
        'my_rates,chemistry_data_storage_ptr' \
        '&my_uvb_rates,photo_rate_storage_ptr' \
        'my_units,code_units_ptr'
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
