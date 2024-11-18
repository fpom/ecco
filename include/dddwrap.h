#include <string>
#include <sstream>
#include <fstream>
#include "ddd/DDD.h"
#include "ddd/util/dotExporter.h"

#define val_t GDDD::val_t

#define ddd_ONE DDD::one
#define ddd_new_ONE() DDD(DDD::one)
#define ddd_new_EMPTY() DDD(DDD::null)
#define ddd_new_TOP() DDD(DDD::top)
#define ddd_new_range(var,val1,val2,d) DDD(var,val1,val2,d)
#define ddd_val_size sizeof(GDDD::val_t)

#define ddd_is_STOP(d) (d == DDD::one) || (d == DDD::null)
#define ddd_is_ONE(d) (d == DDD::one)
#define ddd_is_NULL(d) (d == DDD::null)
#define ddd_is_TOP(d) (d == DDD::top)

#define ddd_concat(a,b) DDD(a ^ b)
#define ddd_union(a,b) DDD(a + b)
#define ddd_intersect(a,b) DDD(a * b)
#define ddd_minus(a,b) DDD(a - b)

#define ddd_iterator GDDD::const_iterator
#define ddd_iterator_begin(d) d.begin()
#define ddd_iterator_next(i) i++
#define ddd_iterator_end(i,d) (i == d.end())
#define ddd_iterator_value(i) i->first
#define ddd_iterator_ddd(i) DDD(i->second)

#define ddd_print(d,s) s << d

#define sdd_ONE GSDD::one
#define sdd_new_ONE() SDD(GSDD::one)
#define sdd_new_EMPTY() SDD(GSDD::null)
#define sdd_new_TOP() SDD(GSDD::top)
#define sdd_is_STOP(s) (s == GSDD::one) || (s == GSDD::null)
#define sdd_is_ONE(s) (s == GSDD::one)
#define sdd_is_NULL(s) (s == GSDD::null)
#define sdd_is_TOP(s) (s == GSDD::top)
#define sdd_concat(a,b) SDD(a ^ b)
#define sdd_union(a,b) SDD(a + b)
#define sdd_intersect(a,b) SDD(a * b)
#define sdd_minus(a,b) SDD(a - b)
#define sdd_eq(a,b) a == b
#define sdd_ne(a,b) a != b

#define sdd_new_SDDs(var,val,s) SDD(var, val, s)
#define sdd_new_SDDd(var,val,s) SDD(var, val, s)

#define sdd_iterator GSDD::const_iterator
#define sdd_iterator_begin(s) s.begin()
#define sdd_iterator_next(i) i++
#define sdd_iterator_end(i,s) (i == s.end())
#define sdd_iterator_sdd(i) SDD(i->second)
#define sdd_iterator_value_is_SDD(i) (dynamic_cast<SDD*>(i->first) != NULL)
#define sdd_iterator_value_is_DDD(i) (dynamic_cast<DDD*>(i->first) != NULL)
#define sdd_iterator_value_is_GSDD(i) (dynamic_cast<GSDD*>(i->first) != NULL)
#define sdd_iterator_SDD_value(i) dynamic_cast<SDD*>(i->first)
#define sdd_iterator_GSDD_value(i) dynamic_cast<SDD*>(i->first)
#define sdd_iterator_DDD_value(i) dynamic_cast<DDD*>(i->first)

#define sdd_print(d,s) s << d

#define shom_new_Shom_null() Shom::null
#define shom_new_Shom_var_ddd(var,val,s) Shom(var,val,s)
#define shom_new_Shom_var_sdd(var,val,s) Shom(var,val,s)
#define shom_neg(h) (!h)
#define shom_eq(a,b) (a == b)
#define shom_ne(a,b) (a != b)
#define shom_call(h,s) SDD(h(s))
#define shom_union(a,b) (a + b)
#define shom_circ(a,b) (a & b)
#define shom_intersect_SDD_Shom(a,b) (a * b)
#define shom_intersect_Shom_SDD(a,b) (a * b)
#define shom_intersect_Shom_Shom(a,b) (a * b)
#define shom_minus_Shom_SDD(a,b) (a - b)
#define shom_minus_Shom_Shom(a,b) (a - b)
#define shom_invert(s,d) s.invert(d)

#define shom_print(h,s) s << h
#define shom_set d3::set<GShom>::type
#define shom_addset(s) GShom::add(s)
