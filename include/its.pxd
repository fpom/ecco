from ddd cimport sdd, shom, Shom, SDD
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.list cimport list
from libcpp.vector cimport vector

cdef extern from "its/Type.hh" namespace "its" :
    cdef cppclass Type :
        ctypedef pair[string,Shom] namedTr_t
        ctypedef list[namedTr_t] namedTrs_t
        Shom observe (vector[string], SDD) const
    ctypedef const Type* pType

cdef extern from "its/Instance.hh" namespace "its" :
    cdef cppclass Instance :
        pType getType()

cdef extern from "its/ITSModel.hh" namespace "its" :
    cdef cppclass ITSModel :
        ITSModel()
        void getNamedLocals (Type.namedTrs_t &) const
        SDD getInitialState()
        SDD computeReachable(bint)
        Shom getNextRel ()
        Shom getPredRel ()
        Instance* getInstance()

cdef class model :
    cdef ITSModel i
    cdef readonly str path, fmt
    cpdef sdd initial (model self)
    cpdef shom succ (model self)
    cpdef shom pred (model self)
    cpdef dict transitions (model self)
