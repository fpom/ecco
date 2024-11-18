/****************************************************************************/
/*								            */
/* This file is part of libDDD, a library for manipulation of DDD and SDD.  */
/*     						                            */
/*     Copyright (C) 2001-2008 Yann Thierry-Mieg, Jean-Michel Couvreur      */
/*                             and Denis Poitrenaud                         */
/*    Based on a file written by Alexandre Duret-Lutz for Spot,             */
/*                                     Alexandre.Duret-Lutz@lip6.fr         */
/*     						                            */
/*     This program is free software; you can redistribute it and/or modify */
/*     it under the terms of the GNU Lesser General Public License as       */
/*     published by the Free Software Foundation; either version 3 of the   */
/*     License, or (at your option) any later version.                      */
/*     This program is distributed in the hope that it will be useful,      */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of       */
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        */
/*     GNU LEsserGeneral Public License for more details.                   */
/*     						                            */
/* You should have received a copy of the GNU Lesser General Public License */
/*     along with this program; if not, write to the Free Software          */
/*Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */
/*     						                            */
/****************************************************************************/
#ifndef __DDD_MISC_HASHFUNC_HH
#define __DDD_MISC_HASHFUNC_HH

/******************************************************************************/

#include <stdint.h>

namespace ddd
{
  /// \addtogroup hash_funcs Hashing functions
  /// \ingroup misc_tools
  /// @{

  /// \brief Thomas Wang's 32 bit hash function.
  ///
  /// Hash an integer amongst the integers.
  /// http://www.concentric.net/~Ttwang/tech/inthash.htm
  inline size_t
  wang32_hash(size_t key)
  {
    // We assume that size_t has at least 32bits.
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
  }

  /// Another of Wang's fast hash with a magic number.
  /// good for (sequence of) integers
  inline uint32_t int32_hash(uint32_t a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
  }


  /// \brief Knuth's Multiplicative hash function.
  ///
  /// This function is suitable for hashing values whose
  /// high order bits do not vary much (ex. addresses of
  /// memory objects).  Prefer spot::wang32_hash() otherwise.
  /// http://www.concentric.net/~Ttwang/tech/addrhash.htm
  inline size_t
  knuth32_hash(size_t key)
  {
    // 2654435761 is the golden ratio of 2^32.  The right shift of 3
    // bits assumes that all objects are aligned on a 8 byte boundary.
    return (key >> 3) * 2654435761U;
  }
  /// @}
}

#endif // DDD_MISC_HASHFUNC_HH
