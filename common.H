/*
Copyright (c) 2014, Ross Evan Johnson All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3) Neither the name of chess-tournament-pairing nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef COMMON_H
#define COMMON_H

#include "int.H"
#include <iostream>
#include <sstream>
#include <assert.h>
#include <stdlib.h>

using namespace std;

#define BYE_ID	0
#define BR	"\n<br />"	/* for debugging output to html */
#define ASSERT(expr)	assert((expr) ? true : (cout << #expr << " FILE: " << __FILE__ << " LINE: " << __LINE__ << BR << endl, false))

#define Min(a,b)	((a)<=(b)?(a):(b))
#define Max(a,b)	((a)>=(b)?(a):(b))
inline string S (char x) { stringstream s; s << x; return s.str(); }
inline string S (unsigned char x) { stringstream s; s << x; return s.str(); }
inline string S (short x) { stringstream s; s << x; return s.str(); }
inline string S (unsigned short x) { stringstream s; s << x; return s.str(); }
inline string S (int x) { stringstream s; s << x; return s.str(); }
inline string S (unsigned x) { stringstream s; s << x; return s.str(); }
inline string S (int64_t x) { stringstream s; s << x; return s.str(); }
inline string S (uint64_t x) { stringstream s; s << x; return s.str(); }
inline string S (float x) { stringstream s; s << fixed << x; return s.str(); }
inline string S (double x) { stringstream s; s << fixed << x; return s.str(); }
inline int I (string s) { return atoi(s.c_str()); }
inline size_t U (string s) { return atoi(s.c_str()); }
inline uint64_t UL (string s) { return atol(s.c_str()); }
inline double F (string s) { return atof(s.c_str()); }

inline size_t FindInvalidUTF8 (const string &s, bool rfc3629, bool debug=false)
{
  // http://en.wikipedia.org/wiki/UTF-8
  // RFC 3629 (Nov 2003) restricts UTF-8 to four bytes (end at U+10FFFF)
  for (size_t x = 0; x < s.size(); ++x) {
    if (debug)
      cout << "x=" << x << " s[x]=" << uint8_t(s[x]) << endl;
    if (0x00 <= uint8_t(s[x]) && uint8_t(s[x]) <= 0x7F						// one byte
	) { x += 0; continue; }
    else if (x+1 < s.size() && (rfc3629?0xC2:0xC0) <= uint8_t(s[x]) && uint8_t(s[x]) <= 0xDF	// two bytes
	&& (uint8_t(s[x+1]) & 0xC0) == 0x80
	) { x += 1; continue; }
    else if (x+2 < s.size() && 0xE0 <= uint8_t(s[x]) && uint8_t(s[x]) <= 0xEF			// three bytes
	&& (uint8_t(s[x+1]) & 0xC0) == 0x80
	&& (uint8_t(s[x+2]) & 0xC0) == 0x80
	) { x += 2; continue; }
    else if (x+3 < s.size() && 0xF0 <= uint8_t(s[x]) && uint8_t(s[x]) <= 0xF7			// four bytes
	&& (uint8_t(s[x+1]) & 0xC0) == 0x80
	&& (uint8_t(s[x+2]) & 0xC0) == 0x80
	&& (uint8_t(s[x+3]) & 0xC0) == 0x80
	) { x += 3;
		if (rfc3629 &&
			((uint8_t(s[x])&0x07)<<18)
			+ ((uint8_t(s[x+1])&0x3F)<<12)
			+ ((uint8_t(s[x+2])&0x3F)<<6)
			+ ((uint8_t(s[x+3])&0x3F)<<0)
			> 0x10FFFF) return x;
		continue; }
    else if (rfc3629)
      return x;
    else if (x+4 < s.size() && 0xF8 <= uint8_t(s[x]) && uint8_t(s[x]) <= 0xFB			// five bytes
	&& (uint8_t(s[x+1]) & 0xC0) == 0x80
	&& (uint8_t(s[x+2]) & 0xC0) == 0x80
	&& (uint8_t(s[x+3]) & 0xC0) == 0x80
	&& (uint8_t(s[x+4]) & 0xC0) == 0x80
	) { x += 4; continue; }
    else if (x+5 < s.size() && 0xFC <= uint8_t(s[x]) && uint8_t(s[x]) <= 0xFD			// six bytes
	&& (uint8_t(s[x+1]) & 0xC0) == 0x80
	&& (uint8_t(s[x+2]) & 0xC0) == 0x80
	&& (uint8_t(s[x+3]) & 0xC0) == 0x80
	&& (uint8_t(s[x+4]) & 0xC0) == 0x80
	&& (uint8_t(s[x+5]) & 0xC0) == 0x80
	) { x += 5; continue; }
    return x;
  }
  return string::npos;
}

inline string SingleQuoted (string s)
{
  string s2 = "'";
  for (unsigned x = 0; x < s.size(); ++x) {
    if (s[x] == '\'')
      s2 += s[x];
    s2 += s[x];
  }
  s2 += "'";
  return s2;
}

inline string NotQuoted (string s)
{ return s; }

#define	NO_BREAK_SPACE	"\u00A0"
inline bool IsBigSpace (const string &s, size_t x)
{ return x+1 < s.size() && s[x] == NO_BREAK_SPACE[0] && s[x+1] == NO_BREAK_SPACE[1]; }

string EraseExtraSpace (string x)
{
  while ((x.size() > 0 && isspace(x[0])) || IsBigSpace(x,0))
    x.erase(0,1+IsBigSpace(x,0));
  ASSERT(x.size() == 0 || !isspace(x[0]));
  while ((x.size() > 0 && isspace(x[x.size()-1])) || IsBigSpace(x,x.size()-2))
    x.erase(x.size()-1-IsBigSpace(x,x.size()-2),1+IsBigSpace(x,x.size()-2));
  ASSERT(x.size() == 0 || !isspace(x[x.size()-1]));
  for (size_t y = 0; y < x.size(); ++y)
    while (IsBigSpace(x,y) ? isspace(x[y+2]) || IsBigSpace(x,y+2) : isspace(x[y]) && (isspace(x[y+1]) || IsBigSpace(x,y+1)))
      x.erase(y, 1 + IsBigSpace(x,y));
  ASSERT(x.size() == 0 || !isspace(x[x.size()-1]));
  return x;
}

string EraseString (string x, string kill)
{
  for (;;) {
    size_t f = x.find(kill);
    if (f == string::npos)
      break;
    x = x.substr(0,f) + x.substr(f+kill.size());
  }
  return x;
}


string MakeName (string n)
{
  //cout << "MakeName(" << n << ")" << endl;
  if (n.size() >= 2 && n[0] == '\"' && n[n.size()-1] == '\"')
    n = n.substr(1,n.size()-2);
  string first, last;
  for (size_t x = 0; x < n.size(); ++x) {
    if (n[x] == ',') {
      const string temp = first;
      first = last;
      last = temp;
      while (x+1 < n.size() && isspace(n[x+1]))
        ++x;
      continue;
    }
    if (x == 0 || isspace(n[x-1]) || n[x-1] == ',')
      first += (islower(n[x]) ? char(n[x]-'a'+'A') : n[x]);  // should be upper
#ifdef OLD_CODE
    else if ((x == 1 || isspace(n[x-2])) && (x+1 >= n.size() || isspace(n[x+1]) || n[x+1] == ','))
      first += n[x];  // no change for two letters (like HS, MS, ES, PS)
#endif /* OLD_CODE */
    else
      first += (isupper(n[x]) ? char(n[x]-'A'+'a') : n[x]);  // should be lower
  }
  first = EraseExtraSpace(first);
  last = EraseExtraSpace(last);
  for (size_t x = 0; x < last.size(); ++x) {
    if (isspace(last[x])) {
      last = last.substr(0,x-1)+NO_BREAK_SPACE+(x+1<last.size()?last.substr(x+1):"");
      x += string(NO_BREAK_SPACE).size()-1;
    }
  }
  string result = EraseExtraSpace(first + ' ' + last);
  //cout << " result=\"" << result << "\"" << endl;
  return result;
}

string SquishName (string n)
{
  ASSERT(string(NO_BREAK_SPACE).size() == 2);
  ASSERT(!isspace(NO_BREAK_SPACE[0]));
  ASSERT(!isspace(NO_BREAK_SPACE[1]));
  n = MakeName(n);
  for (size_t x = 0; x < n.size(); ++x) {
    if (n.substr(x,2) == NO_BREAK_SPACE) {
      n.erase(x,2);
      --x;
    } else if (isspace(n[x])) {
      n.erase(x,1);
      --x;
    }
  }
  return n;
}

#endif /* COMMON_H */
