from .src_model import LineProvider, get_source_regions
from .subroutine_entity import build_subroutine_entity
from .subroutine_sig import (
    SubroutineSignature, ArgDescr, ScalarArrayProp, SubroutineArgRef
)
from .token import Type
from .utils import _valid_fortran_fname

import os

#def _build_interpolate_sig(ndim):
#    assert 1 <= ndim
#    assert ndim <= 5
#
#    input_args, gridPar_args, dgridPar_args = [], [], []
#    args = []
#    for i in range(1, ndim+1):
#        args.append(
#            ArgDescr(f"input{i}", Type.f64, ScalarArrayProp.Scalar())
#        )
#    args.append(
#        ArgDescr("gridDim", Type.i64, ScalarArrayProp.GenericArray(1))
#    )
#    for i in range(1, ndim+1):
#        args += [
#            ArgDescr(f"gridPar{i}", Type.f64, ScalarArrayProp.GenericArray(1)),
#            ArgDescr(f"dgridPar{i}", Type.f64, ScalarArrayProp.Scalar())
#        ]
#    args += [
#        ArgDescr("dataSize", Type.i64, ScalarArrayProp.Scalar()),
#        ArgDescr("dataField", Type.f64, ScalarArrayProp.GenericArray(1)),
#        ArgDescr("value", Type.f64, ScalarArrayProp.Scalar())
#    ]
#    return SubroutineSignature(f"interpolate_{ndim}d_g", tuple(args))

def _build_gaussj_g_sig():
    # we are going to recycle n_ref a few times since it's immutable
    n_ref = SubroutineArgRef(subroutine="gaussj_g", arg="n")
    return SubroutineSignature(
        name=f"gaussj_g",
        arguments=(
            ArgDescr("n", Type.i32, ScalarArrayProp.Scalar()),
            ArgDescr("a", Type.f64, ScalarArrayProp.Array((n_ref, n_ref))),
            ArgDescr("b", Type.f64, ScalarArrayProp.Array((n_ref,))),
            ArgDescr("ierr", Type.i32, ScalarArrayProp.Scalar()),
        )
    )

def _parse_signatures(fname, *, verbose=False):
    if verbose:
        print('->', fname)
    with open(fname, 'r') as f:
        provider = LineProvider(f, fname = fname)
        it = get_source_regions(provider)
        for region in it:
            if not region.is_routine:
                continue
            subroutine = build_subroutine_entity(region, it.prologue)
            if verbose:
                print(f"  |-->{subroutine.name}, "
                      f"n_args: {len(subroutine.arguments)} "
                      f"n_local_vars: {len(subroutine.variables)}")
            yield subroutine.subroutine_signature()

def build_signature_registry(src_dir=None, verbose = False):
    # holds (fname, (sig,...)) pair, where (sig,...) is a tuple of signatures
    # within the fname
    pairs = []
    if pairs is not None:
        with os.scandir(src_dir) as it:
            for entry in it:
                if entry.name == 'gaussj_g.F':
                    signatures = (_build_gaussj_g_sig(),)
                elif _valid_fortran_fname(entry.name):
                    signatures = tuple(
                        _parse_signatures(entry.path, verbose=verbose)
                    )
                else:
                    continue
                pairs.append((entry.path, signatures))
    # sort the order (so that it is deterministic)
    pairs.sort(key=lambda e: e[0])
    out = {}
    for _, sig_tup in pairs:
        for sig in sig_tup:
            out[sig.key] = sig
    return out

    return dict(*pairs)

