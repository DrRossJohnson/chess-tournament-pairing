/*
Copyright (c) 2014 Ross Evan Johnson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef INT_H
#define INT_H

#include <stdint.h>

typedef __int128_t int128_t;  // double words defined for 64-bit GCC
typedef __uint128_t uint128_t;  // double words defined for 64-bit GCC

template <int N>
class Uint {
  private:
    typedef uint128_t UU;
    enum {SIZE=(N+sizeof(UU)*8-1)/sizeof(UU)/8};
    UU u[SIZE];
};

typedef Uint<256> uint256_t;
typedef Uint<512> uint512_t;

using namespace std;
#include <iostream>

//ostream &operator<< (ostream &out, uint128_t u) { return out << real128_t(u); }  // TBD: this changes format
ostream &operator<< (ostream &out, uint128_t u);
//ostream &operator<< (ostream &out, int128_t i) { return out << real128_t(i); }  // TBD: changes format
ostream &operator<< (ostream &out, int128_t i);

#endif /* INT_H */
