from libcpp.map cimport map as cmap
from libcpp.string cimport string

cdef class xdd :
    cdef void _dot (self, str ext, list files)

##
## DDD
##

cdef extern from "dddwrap.h" :
    ctypedef short val_t

cdef extern from "ddd/DDD.h" :
    cdef cppclass DDD :
        cmap[int,string] mapVarName
        @staticmethod
        void varName (int var, const string &name)
        @staticmethod
        const string getvarName (int var)
        DDD ()
        DDD (const DDD &)
        DDD (int var, val_t val, const DDD &d)
        DDD (int var, val_t val)
        bint empty () const
        bint set_equal (const DDD &b) const
        long double set_size () const
        int variable() const
        size_t hash () const
        void pstats (bint reinit)

cdef class ddd (xdd) :
    cdef DDD d
    cpdef str varname (ddd self)
    cpdef ddd pick (ddd self, unsigned int count=*)
    cpdef dict dict_pick (ddd self)
    cpdef tuple vars (ddd self)
    cpdef dict varmap (ddd self)
    cpdef bint stop (ddd self)
    cpdef void print_stats (self, bint reinit=*)
    cpdef void dot (ddd self, str path)
    cpdef ddd drop (ddd self, variables)
    cdef ddd _drop (ddd self, set variables)
    cpdef dict dom (ddd self, dict d=*)
    cpdef str dumps (ddd self)
    cpdef void save (ddd self, str path)
    cpdef to_csv (ddd self, str path)
    cdef _csv (ddd self, str row, DDD head, object out)

cdef ddd makeddd (DDD d)

##
## SDD
##

cdef extern from "ddd/SDD.h" :
    cdef cppclass SDD :
        @staticmethod
        void varName (int var, const string &name)
        @staticmethod
        const string getvarName (int var)
        SDD ()
        SDD (const SDD &)
        SDD (int var, DDD val, const SDD)
        SDD (int var, SDD val, const SDD)
        long double nbStates() const
        bint empty() const
        size_t set_hash() const
        int variable() const
        void pstats (bint reinit)

cdef class sdd (xdd) :
    cdef SDD s
    cpdef str varname (sdd self)
    cpdef sdd pick (sdd self, unsigned int count=*)
    cpdef dict dict_pick (sdd self)
    cpdef tuple vars (sdd self)
    cpdef bint stop (sdd self)
    cpdef void print_stats (self, bint reinit=*)
    cpdef void dot (sdd self, str path)
    cpdef sdd drop (sdd self, variables)
    cdef sdd _drop (sdd self, set variables)
    cpdef str dumps (sdd self)
    cpdef dict varmap (sdd self)

cdef sdd makesdd (SDD s)

##
## Shom
##

cdef extern from "ddd/SHom.h" :
    cdef cppclass Shom :
        Shom ()
        Shom (const SDD &s)
        Shom (const Shom &h)
        Shom fixpoint()
        size_t hash() const

cdef class shom :
    cdef Shom h
    cpdef shom fixpoint (shom self)
    cpdef shom lfp (shom self)
    cpdef shom gfp (shom self)
    cpdef shom invert (shom self, sdd potential)
    cpdef str dumps (shom self)

cdef shom makeshom (Shom h)
