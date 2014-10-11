/*
Copyright (c) 2014, Ross Evan Johnson All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3) Neither the name of chess-tournament-pairing nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* this C++ code finds the best pairings for one section and one round according to USCF rules
 * the greedy algorithm considers all possible swaps of any two players and keeps the best single swap (if better) for each iteration
 * multiple swaps per iteration are considered after no improvements can be made with a single swap, but only if multiple swaps are requested
 * the initial position is one possible pairing, a hint that might yield a better local minimum in this global minimization problem
 * the cost function to be minimized is determined by the USCF Swiss System rules and their priorities as given in the USCF rule book/updates
 */
#include "common.H"
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <iostream>
#include <sstream>
#include <climits>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define MATCH_SWISS_SYS		0	/* make pairings match swiss sys for testing */
#define USE_28N3_0		1	/* Implement variation 28N3 with lowest possible threshold (score=0) so that team blocks in small sections do not impact top players */

#ifdef BETA
#define PERF_DEBUG		1	/* use performance counts */
#define USE_PAIRABLE_COST	1
#else
#define PERF_DEBUG		0
#define USE_PAIRABLE_COST	1
#endif

// map postgres types to C++ types
typedef int64_t bigint;
typedef int32_t integer;
typedef int16_t smallint;
typedef float real;
typedef string text;
typedef char character;
typedef bool boolean;
typedef vector<size_t> sizeVector;
typedef vector<integer> integerVector;
typedef vector<smallint> smallintVector;
typedef vector<real> realVector;
typedef vector<double> doubleVector;
typedef vector<string> StringVector;
typedef vector<bool> BoolVector;
typedef StringVector textVector;
typedef string charVector;

template <class T>
ostream &operator<< (ostream &out, const vector<T> &v)
{
  for (size_t x = 0; x < v.size(); ++x)
    out << (x == 0 ? '{' : ',') << v[x];
  if (v.size() > 0) out << '}';

  return out;
}

/* Crenshaw-Berger tables for Round Robin */
static const string roundRobinPairings[] = {
//	size	round	pairings
"	4	1	1-4 2-3"
,"	4	2	3-1 4-2"
,"	4	3	1-2 3-4"
,"	6	1	3-6 5-4 1-2"
,"	6	2	2-6 4-1 3-5"
,"	6	3	6-5 1-3 4-2"
,"	6	4	6-4 5-1 2-3"
,"	6	5	1-6 2-5 3-4"
,"	8	1	4-8 5-3 6-2 7-1"
,"	8	2	8-7 1-6 2-5 3-4"
,"	8	3	3-8 4-2 5-1 6-7"
,"	8	4	8-6 7-5 1-4 2-3"
,"	8	5	2-8 3-1 4-7 5-6"
,"	8	6	8-5 6-4 7-3 1-2"
,"	8	7	1-8 2-7 3-6 4-5"
,"	10	1	5-10 6-4 7-3 8-2 9-1"
,"	10	2	10-9 1-8 2-7 3-6 4-5"
,"	10	3	4-10 5-3 6-2 7-1 8-9"
,"	10	4	10-8 9-7 1-6 2-5 3-4"
,"	10	5	3-10 4-2 5-1 6-9 7-8"
,"	10	6	10-7 8-6 9-5 1-4 2-3"
,"	10	7	2-10 3-1 4-9 5-8 6-7"
,"	10	8	10-6 7-5 8-4 9-3 1-2"
,"	10	9	1-10 2-9 3-8 4-7 5-6"
};
static const string roundRobinReversals[] = {
//	size	round	windraw	reversals
"	4	3	1	"
,"	4	3	2	4-3"
,"	4	3	3	2-1"
,"	4	3	4	"
,"	6	5	1	5-2 4-3"
,"	6	5	2	4-3"
,"	6	5	3	"
,"	6	5	4	6-1 5-2"
,"	6	5	5	6-1"
,"	6	5	6	"
,"	8	5	1	7-2 5-4"
,"	8	5	2	6-3"
,"	8	5	3	5-4 7-2 2-1"
,"	8	5	4	6-3 3-7 7-2"
,"	8	5	5	8-1 7-4 4-6 6-3"
,"	8	5	6	8-2 5-4"
,"	8	5	7	8-1 6-3"
,"	8	5	8	"
,"	10	7	1	9-2 7-4"
,"	10	7	2	8-3 6-5"
,"	10	7	3	7-4 9-2 2-1"
,"	10	7	4	6-5 8-3 3-9 9-2"
,"	10	7	5	9-2 7-4 2-1 4-8 8-3"
,"	10	7	6	10-2 8-5 5-7 7-4"
,"	10	7	7	10-1 6-5 9-4 4-8 8-3"
,"	10	7	8	10-2 7-4"
,"	10	7	9	10-1 8-3 6-5"
,"	10	7	10	"
};
void CrenshawBergerLookup (
	size_t competitors		// total number of players N
	, size_t round			// current round
	, size_t player			// player number 1 to N
	, size_t withdrawnPlayer	// withdrawn player in the first half (zero for none)
	, integer &board		// assigned board, 1 to N/2
	, character &color)		// assigned color, W or B
{
  //cout << "CrenshawBergerLookup(competitors=" << competitors << ", round=" << round << ", player=" << player << ", withdrawnPlayer=" << withdrawnPlayer << ", board=" << board << ", color=" << color << ")"BR << endl;
  bool isBye = false;
  if (competitors % 2 == 1) {
    ASSERT(withdrawnPlayer == 0);
    isBye = true;
    ++competitors;
  }
  if (withdrawnPlayer == 0)
    withdrawnPlayer = competitors;
  const size_t pSize = sizeof(roundRobinPairings)/sizeof(*roundRobinPairings);
  size_t opponent = 0;
  board = 0;
  color = '*';
  for (size_t x = 0; x < pSize; ++x) {
    //cout << BR << roundRobinPairings[x];
    //	size	round	pairings
    size_t tab = roundRobinPairings[x].find('\t');
    ASSERT(tab != string::npos);
    const size_t size = atoi(roundRobinPairings[x].c_str()+tab+1);
    if (size != competitors) continue;
    //cout << " size=" << size;

    tab = roundRobinPairings[x].find('\t', tab+1);
    ASSERT(tab != string::npos);
    const size_t rnd = atoi(roundRobinPairings[x].c_str()+tab+1);
    if (rnd != round) continue;
    //cout << " rnd=" << rnd;

    tab = roundRobinPairings[x].find('\t', tab+1);
    ASSERT(tab != string::npos);
    string pairings = roundRobinPairings[x].substr(tab+1);
    for (size_t y = 1; ; ++y) {
      const size_t p1 = atoi(pairings.c_str());
      const size_t dash = pairings.find('-');
      ASSERT(dash != string::npos);
      const size_t p2 = atoi(pairings.substr(dash+1).c_str());
      //cout << " p1=" << p1;
      //cout << " p2=" << p2;
      if (p1 == player) {
        //cout << " FOUND";
        ASSERT(opponent == 0 && color == '*');
        opponent = p2;
        board = y;
        color = 'W';
      }
      if (p2 == player) {
        //cout << " FOUND";
        ASSERT(opponent == 0 && color == '*');
        opponent = p1;
        board = y;
        color = 'B';
      }
      const size_t space = pairings.find(' ', dash+1);
      //cout << " space=" << space << endl;
      if (space == string::npos)
        break;
      pairings = pairings.substr(space+1);
    }
  }
  ASSERT(1 <= opponent && opponent != player && opponent <= competitors);
  ASSERT(1 <= board && board <= int(competitors/2));
  ASSERT(color == 'W' || color == 'B');
  //cout << BR"done with pairing" << endl;

  const size_t rSize = sizeof(roundRobinReversals)/sizeof(*roundRobinReversals);
  bool isReversed = false;
  for (size_t x = 0; x < rSize; ++x) {
    //cout << BR << roundRobinReversals[x];
    //	size	round	windraw	reversals
    size_t tab = roundRobinReversals[x].find('\t');
    ASSERT(tab != string::npos);
    const size_t size = atoi(roundRobinReversals[x].c_str()+tab+1);
    if (size != competitors) continue;

    tab = roundRobinReversals[x].find('\t', tab+1);
    ASSERT(tab != string::npos);
    const size_t rnd = atoi(roundRobinReversals[x].c_str()+tab+1);

    tab = roundRobinReversals[x].find('\t', tab+1);
    ASSERT(tab != string::npos);
    const size_t withdraw = atoi(roundRobinReversals[x].c_str()+tab+1);
    if (withdraw != withdrawnPlayer) continue;

    tab = roundRobinReversals[x].find('\t', tab+1);
    ASSERT(tab != string::npos);
    string reversals = roundRobinReversals[x].substr(tab+1);
    if (reversals.size() > 0) for (;;) {
      const size_t p1 = atoi(reversals.c_str());
      const size_t dash = reversals.find('-');
      ASSERT(dash != string::npos);
      const size_t p2 = atoi(reversals.substr(dash+1).c_str());
      if (p1 == player && p2 == opponent) {
        ASSERT(!isBye && withdrawnPlayer != competitors);
        ASSERT(round >= rnd);
        ASSERT(!isReversed);
        color = 'W';
        isReversed = true;
      }
      if (p2 == player && p1 == opponent) {
        ASSERT(!isBye && withdrawnPlayer != competitors);
        ASSERT(round >= rnd);
        ASSERT(!isReversed);
        color = 'B';
        isReversed = true;
      }
      const size_t space = reversals.find(' ', dash+1);
      if (space == string::npos)
        break;
      reversals = reversals.substr(space+1);
    }
  }
  //cout << BR"done with color"BR << endl;
}

struct Player
{
  bigint tmt_id;  // input ignored; used for debugging: tournament number (all players must be the same)
  bigint sec_id;  // input ignored; used for debugging: section number (all players must be the same)
  character trn_type;  // type of tournament: S=swiss, M=match, R=round robin, D=double round robin, 2=double round swiss
  smallint rnd;  // round number 1 to N (all players must have the same round)
  integer board_num;  // input board hint; output final board placement (lowest board is same as input)
  character board_color;  // input color (W=white/bye or B=black) hint; output final color

  integer uscf_id;  // used for debugging: USCF member number
  integer play_id;  // unique identifier for each player (may be USCF member number or other database ID, but not zero)
  text player_name;  // input ignored; used for debugging: player's name
  smallint reentry;  // unique identifier for player reentries (suggest always zero if no reentries)
  integer team_id;  // primary team identifier for prizes and a performance optimization for determining if team block is pairable
  text team_name;  // input ignored; used for debugging: player's primary team name
  integerVector teammates;  // list of player IDs for all teammates (player can have more than one team block; all non-pairing requests (rule 28T) are handled by specifying a team ID, which is not the same as a player ID; if two players have the same team ID in each of their lists, then we avoid pairing them)
  textVector opponents;  // list of opponents already played (in round order); each opponent is a string combining the play_id and reentry separated by an underscore; byes and non-played games are not included

  real score;  // total points from prior rounds (zero for first round)
  smallint rating;  // USCF (or assigned) rating (zero for unrated unless assigned)
  boolean is_unrated;  // whether player is unrated (always use false in an unrated section)
  text use_rating;  // whether section is rated (uscf) or unrated (none)
  smallint provisional; // total number of rated games played before this tournament; may not be in supplement yet, so must get from USCF website or player; 4+ means player already has enough games to get at least a provisional rating, so we worry less about giving a bye to this unrated player in a 4-round event (rules 28L2 & 28L5)
  double rand;  // random value to break ties for ranking players with same score and rating - or for choosing lots on round-robin pairings (must be unique and the same across rounds)
  boolean bye_house;  // house player that should receive bye (half or zero) when odd number of players to pair
  boolean bye_request;  // requested bye (half or zero) for this round?
  smallint unplayed_count;  // total unplayed games for all rounds, full-point assigned (prior rounds), half-point requested (all rounds), and zero-point byes (used for rules 28L2 and 28L5 to determine whether players will have enough games to get a rating)
  smallint half_bye_count;  // total half byes and forfeit wins for all rounds, taken or committed (for rule 28L4)
  smallintVector bye_rounds;  // list of rounds with requested byes (past, current, and future), half point or zero point

  text due_color;  // input ignored; output due color: W=white to equalize, B=black to equalize, w=white to alternate, b=black to alternate, x=neither
  charVector color_history;  // list of assigned colors in prior rounds, W=white, B=black, otherwise neither (f=full point, h=half point, z=zero point); 'f' is used for rule 28L3
  charVector played_colors;  // like color history, but only W and B, not unplayed games
  char first_color;  // color of top player on top board in first round (rules 28J,29E2): W=white, B=black
  smallint multiround;  // number of rounds in a row with same opponent (used to implement multiple games per round)

  boolean paired;  // true if already paired manually (which will not be repaired, but may change board)
  text warn_codes;  // output warning codes (safe to ignore)
  character game_result;  // input ignored; used for debugging: results for current round (blank=unknown)
  integer rank;  // input ignored; used for debugging: player rank by score group and rating starting at 1
  integerVector teammate_ranks;  // input ignored; used for debugging: list of teammate ranks
  integerVector opponent_ranks;  // input ignored; used for debugging: list of prior opponents' ranks
};

ostream &operator<< (ostream &out, const Player &p)
{
  return out
	<< "tmt_id=" << p.tmt_id
	<< " sec_id=" << p.sec_id
	<< " rnd=" << p.rnd
	<< " board_num=" << p.board_num
	<< " board_color=" << p.board_color
	<< " play_id=" << p.play_id
	<< " uscf_id=" << p.uscf_id
	<< " player_name=" << p.player_name
	<< " reentry=" << p.reentry
	<< " team_id=" << p.team_id
	<< " team_name=" << p.team_name
	<< " teammates=" << p.teammates
	<< " opponents=" << p.opponents
	<< " score=" << p.score
	<< " rating=" << p.rating
	<< " is_unrated=" << p.is_unrated
	<< " use_rating=" << p.use_rating
	<< " rand=" << p.rand
	<< " bye_request=" << p.bye_request
	<< " unplayed_count=" << p.unplayed_count
	<< " half_bye_count=" << p.half_bye_count
	<< " bye_rounds=" << p.bye_rounds
	<< " due_color=" << p.due_color
	<< " color_history=" << p.color_history
	<< " played_colors=" << p.played_colors
	<< " first_color=" << p.first_color
	<< " multiround=" << p.multiround
	<< " paired=" << p.paired
	<< " warn_codes=" << p.warn_codes
	<< " game_result=" << p.game_result
	<< " rank=" << p.rank
	<< " teammate_ranks=" << p.teammate_ranks
	<< " opponent_ranks=" << p.opponent_ranks
	;
}

// sort predicate for players (and boards)
bool operator< (const Player &x, const Player &y)
{
  ASSERT(x.rand != y.rand);
  return (x.play_id == BYE_ID) < (y.play_id == BYE_ID) || ((x.play_id == BYE_ID) == (y.play_id == BYE_ID)
	&& (x.bye_request < y.bye_request || (x.bye_request == y.bye_request
	&& (x.paired < y.paired || (x.paired == y.paired
	&& (x.score > y.score || (x.score == y.score
	&& (x.rating > y.rating || (x.rating == y.rating
	&& (x.rand < y.rand || (x.rand == y.rand  // tie breaker for same ratings
	&& (x.play_id < y.play_id || (x.play_id == y.play_id  // handle rare case when random numbers match
	&& (x.reentry < y.reentry  // handle rare case when random numbers match
	))))))))))))));
}
bool operator> (const Player &x, const Player &y)	{ return y < x; }
bool operator!= (const Player &x, const Player &y)	{ return x < y || y < x; }
bool operator== (const Player &x, const Player &y)	{ return !(x != y); }
bool operator<= (const Player &x, const Player &y)	{ return !(x > y); }
bool operator>= (const Player &x, const Player &y)	{ return !(x < y); }

typedef vector<Player> PlayerVector;  // no byes (i.e. play_id != 0)
typedef sizeVector IndexVector;
typedef set<size_t> IndexSet;
const size_t invalidIndex = -1;  // invalid array index

ostream &operator<< (ostream &out, const IndexVector &a)
{
  for (size_t x = 0; x < a.size(); ++x)
    out << (x == 0 ? '{' : ',') << a[x];
  if (a.size() > 0) out << '}';
  return out;
}

typedef int64_t CostValue;
#define MaxCostValue	LLONG_MAX
struct Cost {
  // potential problems in order of significance (most to least)
  // lower values are better (zero is best)
  // comments give relevant USCF pairing rules
  CostValue byeChoice;			// 22C, 29K
  CostValue byeAgain;			// 28L3
  CostValue playersMeetTwice;		// 27A1, 28S1, 28S2, 29C2
  CostValue cantPairPlayers;		// 27A1, 29C2, 29K, 29L
  CostValue teamBlocks2;		// 28N, 28N1, 28T, 29C2
  CostValue unequalScores;		// 27A2, 29A, 29B
  CostValue teamBlocks;			// 28N, 28N1, 28T, 29C2
  CostValue cantPairTeams;		// 28N, 28N1, 28T, 29C2, 29K, 29L
  CostValue byeAfterHalf;		// 28L4
  CostValue lowestScoreBye;		// 28L2, 28L5
  CostValue lowestRatedBye;		// 28L2, 28L5
  CostValue oddPlayerUnrated;		// 29D1
  CostValue oddPlayerMultipleGroups;	// 29D2
  CostValue interchange200;		// 27A3, 29C, 29D, 29E5
  CostValue transpose200;		// 27A5, 29C, 29D, 29E
  CostValue colorImbalance;		// 27A4, 29E4
  CostValue colorRepeat3;		// 29E5f
  CostValue interchange80;		// 27A3, 29D, 29E5
  CostValue transpose80;		// 27A5, 29C, 29D, 29E
  CostValue colorAlternate;		// 27A5
  CostValue interchange0;		// 27A3, 29D, 29E5
  CostValue transpose0;			// 27A5, 29C, 29D, 29E
  CostValue pairingCard;		// 28A, 28B, 29A
  CostValue reversedColors;		// 28J 29E
  CostValue boardOverlap;		// 28J
  CostValue boardOrder;			// 28J
  Cost (void) { ASSERT(MaxCostValue > UINT_MAX); memset(this, 0, sizeof(*this)); }
  size_t players;			// for debugging/printing
  bool IsZero (void) const { Cost z; z.players = this->players; return (memcmp(this, &z, sizeof(z)) == 0); }
};

#define COST_BEGIN	byeChoice
bool operator< (const Cost &c1, const Cost &c2)
{
  for (size_t x = 0; x < sizeof(c1)/sizeof(c1.COST_BEGIN); ++x) {
    if ((&c1.COST_BEGIN)[x] < (&c2.COST_BEGIN)[x])
      return true;
    else if ((&c1.COST_BEGIN)[x] > (&c2.COST_BEGIN)[x])
      return false;
  }
  return false;
}
bool operator> (const Cost &x, const Cost &y)	{ return y < x; }
bool operator>= (const Cost &x, const Cost &y)	{ return !(x < y); }
bool operator<= (const Cost &x, const Cost &y)	{ return !(y < x); }
bool operator== (const Cost &x, const Cost &y)	{ return !(x < y || y < x); }
bool operator!= (const Cost &x, const Cost &y)	{ return !(x == y); }

enum {MAX_RATING=30000+1};  // one more than maximum possible rating
ostream &operator<< (ostream &out, const Cost &c)
{
  bool found = false;
  int num = 0;
#define O(v)	{ ++num; if (c.v != 0) { out << (found?" ":"") << num << ")"#v"=" << c.v; found = true; } }
#define ON(v)	{ if (c.v != 0) { out << (found?" ":"") << #v"=" << c.v; found = true; } }
#define OP(v)	{ ++num; if (c.v != 0) { out << (found?" ":"") << num << ")"#v"=" << c.v / (MAX_RATING*c.players) << ',' << c.v % (MAX_RATING*c.players); found = true; } }
  O(byeChoice)
  O(byeAgain)
  O(playersMeetTwice)
  O(cantPairPlayers)
  O(teamBlocks)
  O(cantPairTeams)
  O(unequalScores)
  O(byeAfterHalf)
  O(lowestScoreBye)
  O(lowestRatedBye)
  O(oddPlayerUnrated)
  O(oddPlayerMultipleGroups)
  OP(interchange200)
  OP(transpose200)
  O(colorImbalance)
  O(colorRepeat3)
  OP(interchange80)
  OP(transpose80)
  O(colorAlternate)
  OP(interchange0)
  OP(transpose0)
  O(pairingCard)
  O(reversedColors)
  out << (found ? ";" : "zero; ");
  ON(players)
#undef O
#undef O1
  return out;
}

// pl array may be resorted by rank after recomputing ranks
// totalRounds = total number of rounds (may use round-robin-like pairings for small swiss)
// firstBoardNum is the number of the top board; if zero, program will make a guess
Cost FindPairings(PlayerVector &pl, smallint totalRounds, integer firstBoardNum, int depth, bool useFirstPairings, bool skipOptimize, const string &secName);

////////////////////////  IMPLEMENTATION  ////////////////////////

////////////////////////  COST FUNCTIONS  ////////////////////////

StringVector costDescription;

void CostDescription (string &warn_codes, char wCode, const char *desc)
{
  if (wCode != 0) {
    enum {MAX_CODES=26*2};
    const char wCodeN = (wCode <= 'Z' ? wCode-'A' : 26+wCode-'a');
    static BoolVector init(MAX_CODES+1, false);
    if (!init[MAX_CODES]) {
      costDescription = StringVector(MAX_CODES);
      init[MAX_CODES] = true;
    }
    ASSERT(0 <= wCodeN && wCodeN < MAX_CODES);
    if (!init[wCodeN]) {
      costDescription[wCodeN] = desc;
      init[wCodeN] = true;
    }
    if (warn_codes.find(wCode) == string::npos)
      warn_codes += wCode;
  }
}

CostValue Multiple (CostValue cv, size_t players, char wCode)
{
  if (pow(players, cv) > MaxCostValue)
    cout << "Multiple(cv=" << cv << ",players=" << players << ",wCode=" << wCode << '(' << int(wCode) << ')' << ") may be too large"BR << endl;
  CostValue result = 0;
  for (CostValue x = 0; x < cv; ++x) {
    CostValue temp = result;
    result += pow(players, x);
    if (result < temp)
      result = MaxCostValue;
  }
  return result;
}

CostValue ByeChoice (char wCode, Player &x, const Player &y)
{
  // rules 22C, 28M1, 29K
  CostValue cv = 0;
  if (x.play_id != BYE_ID
	&& !x.bye_house		// rule 28M1 - house player should receive bye instead of others
	&& (x.bye_request ?
	y.play_id != BYE_ID :	// rule 22C - otherwise forfeit loss will deprive opponent of game
	y.play_id == BYE_ID)	// rule 29K,L - players prefer rematches over byes
	) {
    //cout << "x: " << x << BR << endl;
    //cout << "y: " << y << BR << endl;
    cv = 1;
  }
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Bye request mismatch (22C,28M1,29K)");
  return cv;
}

CostValue ByeAgain (char wCode, Player &x, const Player &y, size_t players)
{
  // rule 28L3
  CostValue cv = 0;
  if (x.play_id != BYE_ID && y.play_id == BYE_ID) {
    const size_t cnt = count(x.color_history.begin(), x.color_history.end(), 'f');
    cv = Multiple(cnt, players, wCode);
  }
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Bye ineligible (28L3)");
  return cv;
}

char SameColor (char color)
{ return toupper(color) == 'W' ? 'W' : toupper(color) == 'B' ? 'B' : 'x'; }
char FlipColor (char color)
{ return toupper(color) == 'W' ? 'B' : toupper(color) == 'B' ? 'W' : 'x'; }

CostValue IdenticalMatch (char wCode, Player &x, const Player &y, size_t players, char xColor)
{
  CostValue rematchX = 0, rematchY = 0;
  for (size_t z = 0; z < x.opponents.size(); ++z)
    if (x.opponents[z] == S(y.play_id)+"_"+S(y.reentry) && x.played_colors[z] == xColor)
      ++rematchX;
  for (size_t z = 0; z < y.opponents.size(); ++z)
    if (y.opponents[z] == S(x.play_id)+"_"+S(x.reentry) && y.played_colors[z] == FlipColor(xColor))
      ++rematchY;
  const CostValue rematch = max(rematchX, rematchY);
  const CostValue cv = Multiple(rematch, players, wCode);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "IdenticalMatch");
  return cv;
}

CostValue PlayersMeetTwice (char wCode, Player &x, const Player &y, size_t players)
{
  // rules 27A1, 28S1, 28S2, 29C2
  CostValue rematchX = 0, rematchY = 0;
  for (size_t z = 0; z < x.opponents.size(); ++z)
    if (I(x.opponents[z]) == y.play_id)  // I() strips reentry from each x.opponents[z]
      ++rematchX;
  for (size_t z = 0; z < y.opponents.size(); ++z)
    if (I(y.opponents[z]) == x.play_id)  // I() strips reentry from each y.opponents[z]
      ++rematchY;
  const CostValue rematch = max(rematchX, rematchY);
  const CostValue cv = Multiple(rematch, players, wCode);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Players meet twice (27A1,28S1,28S2,29C2)");
  return cv;
}

typedef vector<int> GridElem;  // represent round number of pairing or bye (sometimes -1 for past rounds)
typedef vector<GridElem> PairGrid;
typedef vector<GridElem> ByeGrid;

ostream &operator<< (ostream &out, const PairGrid pg)
{
  out << "<TABLE border=1>\n";
  out << "<TR><TD></TD>";
  for (size_t x = 0; x < pg.size(); ++x)
    out << "<TD>" << x+1 << "</TD>";
  out << "</TR>\n";
  for (size_t x = 0; x < pg.size(); ++x) {
    out << "<TR><TD>" << x+1 << "</TD>";
    for (size_t y = 0; y < pg[x].size(); ++y)
       out << "<TD>" << pg[x][y] << "</TD>";
    out << "</TR>\n";
  }
  out << "</TABLE>\n";
  return out;
}

// color[x] goes with pair[x], not necessarily pl[pair[x]]
// grid: upper triangle is next round pairings, lower is all past rounds
// players is the size of the 2-D square grid
// rounds is number of rounds remaining (counting current)
// byes[X][Y] is whether player X has bye in future round Y from end
// begin and end is the range of rows that are used for this iteration through players
//	begin relates to number of pairings already made
//	end relates to number of pairings remaining to be made
bool Pairable (PairGrid &grid, int rounds, const ByeGrid &bye);
bool Pairable (PairGrid &grid, int rounds, const ByeGrid &bye, int begin, int end)
{
  //cout << "Pairable() WIP" << endl;
  //cout << "rounds=" << rounds << " begin=" << begin << " end=" << end << BR << endl;
  //return true;
  const int players = grid.size();
  if (players <= 1)
    return true;
  if (players < end)
    cout << "parameters to Pairable(players=" << players << ",end=" << end << ") may not be calculated right"BR << endl;
  for (int row = begin; row < end && row < players; ++row) {
    if (bye[row][rounds-1])
      continue;
    for (int col = row + 1; col < players; ++col) {
      if (bye[col][rounds-1])
        continue;
      if (grid[row][col] || grid[col][row])
        continue;
      for (int z = 0; z < row; ++z)
        if (grid[z][col] || grid[z][row])
          goto nextCol;
      grid[row][col] = rounds;  // try this pairing
      if (end >= players) {
        // check next round
        if (rounds <= 1) {
          //cout << grid;
          return true;  // found it
        }
        PairGrid newGrid = grid;
        for (int x = 0; x < players-1; ++x) {
          for (int y = x+1; y < players; ++y) {
            if (grid[x][y])
              newGrid[y][x] = rounds;
            newGrid[x][y] = 0;
          }
        }
        const bool isFound = Pairable(newGrid, rounds-1, bye);
        if (isFound) {
          grid = newGrid;
          return true;
        }
      } else {  // need more pairings this round
        // check next pairing this round
        const bool isFound = Pairable(grid, rounds, bye, row+1, end+1);
        if (isFound)
          return true;
      }
      grid[row][col] = 0;  // this pairing didn't work
      nextCol:;
    }
  }
  return false;
}

bool Pairable (PairGrid &grid, int rounds, const ByeGrid &bye)
{
  if (rounds <= 0) return true;
  ASSERT(rounds > 0);
  const size_t players = grid.size();
  size_t byes = 0;
  for (size_t x = 0; x < players; ++x)
    byes += bye[x][rounds-1];
  return Pairable(grid, rounds, bye, 0, players-(players-byes)/2+1);
}

bool IsOneTeamMajority (const PlayerVector &pl)
{
  ASSERT(pl.size() > 0 && pl.back().play_id == BYE_ID);
  integerVector team(pl.size()-1);
  for (size_t x = 0; x < pl.size()-1; ++x) {
    ASSERT(pl[x].play_id != BYE_ID);
    team[x] = pl[x].team_id;
  }
  sort(team.begin(), team.end());
  int mode=0, next=0;
  size_t modeCnt=0, nextCnt=0;
  for (size_t x = 0; x < team.size(); ++x) {
    if (team[x] == next) {
      ++nextCnt;
    } else {
      next = team[x];
      nextCnt = 1;
    }
    if (nextCnt > modeCnt) {
      mode = next;
      modeCnt = nextCnt;
    }
  }
  // use >= rather than > because experiments show that exactly half the size is a performance problem
  const bool isOneTeamMajority = (mode != 0 && 2 * modeCnt >= team.size());
#if PERF_DEBUG
  if (isOneTeamMajority) {
    static bigint lastSec = 0;
    if (lastSec != pl[0].sec_id) {
      lastSec = pl[0].sec_id;
      cout << "sec_id=" << lastSec << ": IsOneTeamMajority()=true"BR << endl;
    }
  }
#endif
  return isOneTeamMajority;
}

#if USE_PAIRABLE_COST
CostValue PairableCost (char wCode, PlayerVector &pl, const IndexVector &pair, size_t remainingRounds, bool isTeam)
{
  //cout << "PairableCost(remainingRounds=" << remainingRounds << ",isTeam=" << isTeam << ")"BR << endl;
  //cout << pl << BR << endl;

  // rules 27A1, 29C2, 29K, 29L - avoid meeting twice in future rounds by using (something like) round robin pairings
  // also rules 28N, 28N1, 28T when isTeam=true
  // instead of using round-robin (RR) pairing tables, this function provides a blend of RR and Swiss such that
  // RR pairings occur as number of rounds approaches number of players,
  //	but RR pairings may not match published RR tables since this function invents new RR pairings as needed
  // Swiss flexiblity is maintained: approximate RR tables are invented as players
  //	withdraw, register late, request byes, or request non-pairings
  // TBD: In complicated situations, this function might find 1 vs 2 pairings (rule 29L1) instead of the best pairing for top board (rule 29L)
  //	This could be improved by changing the exhaustive search order to first search the best top board pairing

  // calculate pairable on last player in each section
  if (remainingRounds <= 0)
    return 0;
  if (isTeam && IsOneTeamMajority(pl))
    return 1;
  size_t rounds = pl[0].rnd + remainingRounds;
  size_t num = pl.size() - 1;  // number of non-bye players
  ByeGrid bye;
  PairGrid pg;
  bye.reserve(num);
  pg.reserve(num);
  for (size_t y = 0; y < num; ++y) {
    bye.push_back(GridElem(remainingRounds,0));
    pg.push_back(GridElem(num,0));
    pg.at(y).at(y) = -11;
  }
  // put opponents and teammates in lower triangle - and record byes
  //cout << "before triangle num=" << num << BR << endl;
  //cout << pg << endl;
  for (size_t y = 0; y < num; ++y) {
    const size_t r1 = pl[y].rank;
    //cout << " y=" << y << " r1=" << r1 << BR << endl;
    if (r1 >= num) {
      cout << "Pairable() inputs problem in PairableCost()" << endl;
      continue;
    }
    const smallintVector &b = pl[y].bye_rounds;
    //cout << " b=" << b << BR << endl;
    for (size_t z = 0; z < b.size(); ++z) {
      const size_t rnd = b[z];
      //cout << "r1=" << r1 << " rounds=" << rounds << " rnd=" << rnd << BR << endl;
      if (rnd > rounds)
        cout << "invalid bye round=" << rnd << " for r1=" << r1 << " in PairableCost()" << endl;
      else if (rounds-rnd < remainingRounds)
        bye.at(r1).at(rounds-rnd) = 1;
    }
#ifdef OLD_CODE
    const size_t r2 = pl[y].rank;
    //cout << " r2=" << r2 << BR << endl;
    if (r2 < num) {
      if (r1 < r2)
        pg.at(r2).at(r1) = -1;
      else
        pg.at(r1).at(r2) = -1;
    } else {
      // gets here only for byes?
      cout << "r1=" << r1 << " r2=" << r2 << BR << endl;
    }
#endif /* OLD_CODE */
    const integerVector o = pl[y].opponent_ranks;
    //cout << "o=" << o << BR << endl;
    for (size_t z = 0; z < o.size(); ++z) {
      const size_t r2 = o[z];
      if (r2 >= num)
        continue;
      if (r1 < r2)
        pg.at(r2).at(r1) = -1;
      else
        pg.at(r1).at(r2) = -1;
    }
    if (isTeam) {
      const integerVector t = pl[y].teammate_ranks;
      //cout << "t=" << t << BR << endl;
      for (size_t z = 0; z < t.size(); ++z) {
        const size_t r2 = t[z];
        if (r2 >= num)
          continue;
        if (r1 < r2)
          pg.at(r2).at(r1) = -1;
        else
          pg.at(r1).at(r2) = -1;
      }
    }
  }
  // also record current pairings, not just historical pairings from above
  ASSERT(pair.size() % 2 == 0);
  for (size_t y = 0; y < pair.size(); y += 2) {
    const size_t r1 = pair[y];
    const size_t r2 = pair[y+1];
    ASSERT(r1 != r2);
    if (pl[r1].play_id != BYE_ID && pl[r2].play_id != BYE_ID) {
      if (r1 < r2)
        pg.at(r2).at(r1) = -1;
      else
        pg.at(r1).at(r2) = -1;
    }
  }
  //cout << "PairableCost(): before Pairable()"BR << endl;
  //cout << pg;
  const bool isPairable = Pairable(pg, remainingRounds, bye);
  //cout << "PairableCost(): after Pairable()"BR << endl;
  //cout << pg;
  if (!isPairable) CostDescription(pl[0].warn_codes, wCode, (isTeam ? "Can't pair future rounds with team block (28N,U)" : "Can't pair future rounds (27A1)"));
  //cout << "end PairableCost()"BR << endl;
  return !isPairable;
}
#endif /* USE_PAIRABLE_COST */

#if !USE_28N3_0
CostValue TeamBlocks2 (char wCode, Player &x, const Player &y, size_t players)
{
  // rules 28N, 28N1, 28T
  // this is split into two functions before and after UnequalScores() to implement rule 28N1
  // this half implements persons that don't have at least a plus-two score
  CostValue team = 0;
#define PLUS_SCORE(z)	(z.score - (z.rnd/2.0))
  if (x.rank < y.rank && (PLUS_SCORE(x) < 2 || PLUS_SCORE(y) < 2))  // rule 28N1
    for (size_t z = 0; z < x.teammates.size(); ++z)
      if (x.teammates[z] == y.play_id)
        ++team;
#undef PLUS_SCORE
  const CostValue cv = Multiple(team, players, wCode);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Team block violated, not plus-two (28N,U)");
  return cv;
}
#endif /* !USE_28N3_0 */

CostValue UnequalScores (char wCode, Player &x, const Player &y, size_t players, size_t remainingRounds)
{
  // rules 27A2, 29A, 29B
  //const size_t rounds = x.rnd + remainingRounds;
  //return x.rank < y.rank ? Multiple(round(2 * rounds * 2 * Max(x.score,y.score) + 2 * fabs(x.score-y.score)), players, wCode) : 0;
  //const CostValue cv = (x.score != y.score && x.rank < y.rank ? round(2 * fabs(x.score-y.score) * players * (x.rnd+1) + 2 * Max(x.score,y.score)) : 0);
  const CostValue cv = (x.score != y.score && x.rank < y.rank ? round(Multiple(2*fabs(x.score-y.score), x.rnd, wCode) * x.rnd + 2 * Max(x.score,y.score)) : 0);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Unequal scores (27A2,29A,29B)");
  return cv;
}

CostValue TeamBlocks (char wCode, Player &x, const Player &y, size_t players)
{
  // rules 28N, 28N1, 28T
  // this is split into two functions before and after UnequalScores() to implement rule 28N1
  // this half implements all persons (including those with plus-two score)
  CostValue team = 0;
  if (x.rank < y.rank)
    for (size_t z = 0; z < x.teammates.size(); ++z)
      if (x.teammates[z] == y.play_id)
        ++team;
  const CostValue cv = Multiple(team, players, wCode);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Team block violated (28N,U)");
  return cv;
}

CostValue ByeAfterHalf (char wCode, Player &x, const Player &y, size_t players)
{
  // rule 28L4
  const CostValue cv = (x.play_id != BYE_ID && y.play_id == BYE_ID && !x.bye_request ?
		Multiple(x.half_bye_count, players, wCode) :
		0);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Bye after half (28L4)");
  return cv;
}

CostValue LowestScoreBye (char wCode, Player &x, const Player &y, size_t players, real lowestScore)
{
  // rule 28L2; (28L5 not yet implemented)
  // lowest rated is handled by interchange and transpose
  CostValue cv = 0;
  if (x.play_id != BYE_ID && y.play_id == BYE_ID && !x.bye_request && x.score - lowestScore > 0.25)
    cv = Multiple(2*(x.score-lowestScore), players, wCode);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Bye player is not from the lowest score group (28L2)");
  return cv;
}

CostValue LowestRatedBye (char wCode, Player &x, const Player &y, size_t remainingRounds)
{
  // rule 28L2; (28L5 not yet implemented)
  // lowest rated is handled by interchange and transpose
  CostValue cv = 0;
  if (x.play_id != BYE_ID && y.play_id == BYE_ID && !x.bye_request && x.is_unrated && x.use_rating != "none") {
    if (x.provisional + (x.rnd + remainingRounds - x.unplayed_count - 1) < 4)
      cv = 2;
    else
      cv = 1;
  }
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Bye player unrated and (if cost=2) may have too few games (28L2)");
  return cv;
}

CostValue OddPlayerUnrated (char wCode, Player &x, const Player &y)
{
#if MATCH_SWISS_SYS
  return 0;
#endif
  // rule 29D1
  // lowest score/rated is handled by interchange and transpose
  const CostValue cv = (x.play_id != BYE_ID && y.play_id != BYE_ID && x.score != y.score && x.is_unrated && x.use_rating != "none");
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Odd player unrated (29D1)");
  return cv;
}

CostValue OddPlayerMultipleGroups (char wCode, Player &x, const Player &y, size_t players)
{
  // rule 29D2
  // lowest score/rated is handled by interchange and transpose
  // multiple score groups shouldn't be flagged if the point difference is greater than 0.5,
	// but it won't make a difference in the optimization, so this is simpler
  // the rules are ambiguous on whether a combined 0.5 and 1.5 point drop is preferred to a 1.0 and 1.0 when two players are dropped
	// this prefers the 1.0 and 1.0 case, but the situation would be rare
  const CostValue cv = (x.play_id != BYE_ID && y.play_id != BYE_ID && x.score - y.score > 0.75 ?
			Multiple(2*(x.score-y.score-0.5), players, wCode) :
			0);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Odd player across multiple groups (29D2)");
  return cv;
}

// determine due color based on rule 29E
// upper case means equalization, lower case means alternation, 'x' means neither
// if multiround, only consider first in series against opponent
string DueColor (text history, smallint multiround)
{
  if (multiround != 1 && history.size() > 0) {
    ASSERT(multiround > 0 && history.size() % multiround == 0);
    text h2;
    for (size_t x = 0; x < history.size(); x += multiround)
      h2 += history[x];
    history = h2;
  }
  size_t unplayed = 0;
  for (size_t x = 0; x < history.size(); ++x)
    if ('a' <= history[x] && history[x] <= 'z')
      ++unplayed;
  if (unplayed == history.size())	return "x";
  const size_t whites = count(history.begin(), history.end(), 'W');
  const size_t blacks = count(history.begin(), history.end(), 'B');
  ASSERT(whites + blacks + unplayed == history.size());
  if (whites > blacks)			return string(whites-blacks, 'B');
  else if (blacks > whites)		return string(blacks-whites, 'W');
  for (size_t x = history.size(); x > 0; --x)
    if (history[x-1] == 'W' || history[x-1] == 'B')
      return string(1, char(tolower(FlipColor(history[x-1]))));
  return "x";  // shouldn't get here
}

// assign colors based on rules 28J and 29E2,4 and 30F
char AllocateColor (const Player &x, const Player &y, bool isOddBoard)
{
  //cout << x << BR << y << BR << endl;

  // player with bye gets white; bye gets black
  //cout << 1 << BR << endl;
  //if (x.play_id == BYE_ID || y.play_id == BYE_ID)
    //cout << "x: " << x << BR << "y: " << y << BR << endl;
  if (y.play_id == BYE_ID) return 'W';
  else if (x.play_id == BYE_ID) return 'B';

  // both sides not due any color; rules 28J & 29E2: first round color
  const bool isUpper = (x < y);
  //cout << 2 << BR << endl;
  if (x.due_color == "x" && y.due_color == "x") {
    const char c = ((isUpper ? isOddBoard : !isOddBoard) ? SameColor : FlipColor)(x.first_color);
    //cout << "isUpper=" << isUpper << " isOddBoard=" << isOddBoard << " c=" << c << BR << endl;
    return c;
  }

  // if prior matches against this opponent, then equalize color against this opponent (30F)
  CostValue matchCountWhite = 0, matchCountBlack = 0;
  //size_t rematchIndex = 0;
  for (size_t z = 0; z < x.opponents.size(); ++z) {
    if (x.opponents[z] == S(y.play_id)+"_"+S(y.reentry)) {
      if (toupper(x.played_colors[z]) == 'W')
        ++matchCountWhite;
      else if (toupper(x.played_colors[z]) == 'B')
        ++matchCountBlack;
      //++rematchCount;
      //rematchIndex = z;
    }
  }
  if (matchCountWhite < matchCountBlack)
    return 'W';
  else if (matchCountBlack < matchCountWhite)
    return 'B';
  //if (rematchCount % 2 == 1)
    //return FlipColor(x.played_colors[rematchIndex]);

  // one side not due any color or both get due colors
  //cout << 3 << BR << endl;
  if (y.due_color == "x")
    return SameColor(x.due_color[0]);  // x gets due color
  else if (x.due_color == "x")
    return FlipColor(y.due_color[0]);  // y gets due color
  else if (SameColor(y.due_color[0]) != SameColor(x.due_color[0]))
    return SameColor(x.due_color[0]);  // both get due color

  // equalization of colors takes priority
  //cout << 4 << BR << endl;
  if (isupper(x.due_color[0]) && (!isupper(y.due_color[0]) || x.due_color.size() > y.due_color.size()))
    return SameColor(x.due_color[0]);  // x gets due color
  else if (isupper(y.due_color[0]) && (!isupper(x.due_color[0]) || y.due_color.size() > x.due_color.size()))
    return FlipColor(y.due_color[0]);  // y gets due color

  // most-recent unequal color history breaks ties (opposite of color that does not match)
  //cout << 5 << BR << endl;
  ASSERT(x.color_history.size() == y.color_history.size());
  for (size_t z = x.color_history.size(); z > 0; --z)
    if (SameColor(x.color_history[z-1]) != SameColor(y.color_history[z-1]))  // rule 29E4.4
      return SameColor(x.color_history[z-1]) == 'x' ? SameColor(y.color_history[z-1]) : FlipColor(x.color_history[z-1]);

  // finally, use rank to break ties
  //cout << 6 << BR << endl;
  return x.rank < y.rank ? SameColor(x.due_color[0]) : FlipColor(y.due_color[0]);  // rule 29E4.5
}

CostValue ColorImbalance (char wCode, Player &x, const Player &y, char xColor)
{
  // rules 27A4, 29E4
  //cout << "ColorImbalance()"BR << endl;
  const CostValue cv = (x.due_color[0] == toupper(x.due_color[0]) && xColor != x.due_color[0] && x.play_id != BYE_ID && y.play_id != BYE_ID);
#ifdef OLD_CODE
	x.due_color == y.due_color && (x.due_color == "W" || x.due_color == "B")
	&& x.rank < y.rank
#endif /* OLD_CODE */
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Color not balanced (27A4)");
  return cv;
}

CostValue ColorRepeat3 (char wCode, Player &x, const Player &y, char xColor)
{
  // rule 29E5f
  //cout << "ColorRepeat3()"BR << endl;
  if (x.play_id == BYE_ID || y.play_id == BYE_ID)
    return 0;
  const char yColor = FlipColor(xColor);
  int count = 1;
  for (size_t z = x.color_history.size(); z > 0; --z) {
    if (x.color_history[z-1] == xColor)
      ++count;
    else if (x.color_history[z-1] == yColor)
      break;
  }
  const CostValue cv = (count >= 3);
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Color 3+ in a row (29E5f)");
  return cv;
}

CostValue ColorAlternate (char wCode, Player &x, const Player &y, char xColor)
{
  // rule 27A5
  //cout << "ColorAlternate()"BR << endl;
  if (x.play_id == BYE_ID || y.play_id == BYE_ID)
    return 0;
  //const char yColor = FlipColor(xColor);
  CostValue cv = 0;
  if (xColor != toupper(x.due_color[0])) {
    for (size_t z = x.color_history.size(); z > 0; --z) {
      if ('a' <= x.color_history[z-1] && x.color_history[z-1] <= 'z')
        continue;
      cv = (x.color_history[z-1] == xColor);
      break;
    }
  }
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Color not alternating (27A5)");
  return cv;
}

CostValue Interchange (char wCode, Player &x, const Player &y, size_t players, smallint medianRating, smallint highestRating, smallint unratedRating, size_t threshold)
{
  //cout << "Interchange(" << wCode << ",x,y," << players << ',' << medianRating << ',' << unratedRating << ',' << threshold << ")"BR << endl;
  //cout << x << BR << y << BR << endl;
  // rules 27A3, 29C, 29D, 29E5
  const int dl = threshold;
  const int r0 = x.rating;
  const int r1 = (x.is_unrated && x.use_rating != "none" ? unratedRating : x.rating);
  const int r1u = (x.is_unrated && x.use_rating != "none" && threshold != 0 ? MAX_RATING : r1);
  const int r2 = y.rating;
  const int rm = medianRating;
  const int rh = highestRating;
  CostValue cv;
  if (x.play_id == BYE_ID) {
    cv = 0;
  } else if (y.play_id == BYE_ID) {
    //cout << "shouldn't be above the median (rule 28L2)"BR << endl;
    cv = (rm + dl < r1 ? players * MAX_RATING + r1 - rm : 0);
  } else if (x.score == y.score && x.rank > y.rank && rm + dl < Min(r0,r2)) {
    //cout << "both players above median"BR << endl;
    cv = players * MAX_RATING + Min(r0,r2) - rm;
  } else if (false && x.score < y.score && r1 < rh - dl) {
    //cout << "player pulled up is not highest (not considering unrated, 29E5g)"BR << endl;
    cv = players * MAX_RATING + rh - r1;  // treat transposition like interchange because playing higher rated instead of lower rated (NO: rules 29D2 and 29E5 say this is a transposition)
  } else if (false && x.score < y.score && r1u < rh - dl) {
    //cout << "player pulled up is not highest (not considering unrated, 29E5g)"BR << endl;
    cv = players * MAX_RATING + rh - r1u;  // treat transposition like interchange because playing higher rated instead of lower rated (NO: rules 29D2 and 29E5 say this is a transposition)
  } else if (x.score < y.score && r0 + dl < rm) {
    //cout << "player pulled up is below median"BR << endl;
    cv = players * MAX_RATING + rm - r0;
  } else if (x.score > y.score && rm + dl < r0) {
    //cout << "player dropped down is above median"BR << endl;
    cv = players * MAX_RATING + r0 - rm;
  } else {
    cv = 0;
  }
  if (cv != 0) {
    CostDescription(x.warn_codes, wCode, (
		threshold >= 200 ? "Interchange above 200 (27A3;29E5b,e,g)" :
		threshold >= 80  ? "Interchange above 80 (27A3;29E5b,e,g)" :
				"Interchange above 0 (27A5)"));
    //cout << "x: " << x << BR << endl;
    //cout << "y: " << y << BR << endl;
    //cout << "medianRating=" << medianRating << " threshold=" << threshold << BR << endl;
    //cout << "Interchange: dl=" << dl << " r1=" << r1 << " r1u=" << r1u << " r2=" << r2 << " rm=" << rm << " cv=" << cv << BR << endl;
  }
  return cv;
}

CostValue Transpose (char wCode, PlayerVector &pl, const IndexVector &pair, size_t x, size_t y, smallint unratedRating, size_t threshold, size_t pBegin, size_t pEnd)
{
  //cout << "Transpose(" << pl.size() << ',' << pair.size() << ',' << x << ',' << y << ',' << unratedRating << ',' << threshold << ',' << pBegin << ',' << pEnd << ")"BR << endl;
  // rules 27A5, 29C, 29D, 29E
  // just compare the bottom or highest rank (top is already sorted) and only look downward (others will compare down from above)
  ASSERT(pBegin % 2 == 0 && pEnd % 2 == 0);
  ASSERT(0 <= pBegin && pBegin < pEnd && pEnd <= pair.size());
  ASSERT(pBegin <= x && x < pEnd && pBegin <= y && y < pEnd);
  const size_t players = pl.size();
  Player &px = pl[pair[x]];
  const Player &py = pl[pair[y]];
  //if (threshold == 0 && (px.rank==23-1 || px.rank==25-1)) {
    //cout << "px: " << px << BR << endl;
    //cout << "py: " << py << BR << endl;
  //}
  if (px.play_id == BYE_ID || py.play_id == BYE_ID)
    return 0;
  ASSERT(x%2 == 0 ? y == x+1 : y == x-1);
  ASSERT(x%2 == 0 ? px.rank < py.rank : px.rank > py.rank);
  CostValue cv;
  //if (threshold == 0 && (px.rank==23-1 || px.rank==25-1))
    //cout << "Transpose: threshold=" << threshold << " px.rank=" << px.rank << " py.rank=" << py.rank << " pair[x]=" << pair[x] << " pair[y]=" << pair[y] << BR << endl;
  if (px.rank < py.rank || (false && px.is_unrated && px.use_rating != "none" && threshold != 0)) {
    cv = 0;
  } else {
    ASSERT(px.rank > py.rank);  // px is lower half or pull up
    ASSERT(x % 2 == 1);
    const float sx = px.score;
    const float sy = py.score;
//#define IS_UNRATED(player)	((player).is_unrated && (player).use_rating != "none")
#define IS_UNRATED(player)	((player).is_unrated && (player).use_rating[0] != 'n')  /* performance enhancement? since Transpose() is slow */
    const int rx = (IS_UNRATED(px) ? unratedRating : px.rating);  // rules 29E5g & 29E5 TD TIP
    const int ry = (IS_UNRATED(py) ? unratedRating : py.rating);
    const int kx = px.rank;
    //const int ky = py.rank;
    const int dl = threshold;
    //if (threshold == 0 && (px.rank==23-1 || px.rank==25-1))
      //cout << " sx=" << sx << " sy=" << sy << " rx=" << rx << " ry=" << ry << " kx=" << kx << " dl=" << dl << BR << endl;
    //int rm = -1;  // maximum rating for potential transposition
    for (size_t z = x + 1; z < pEnd; z += 2) {
      //cout << "z=" << z << " pair[z]=" << pair[z] << " pair[z+1]=" << pair[z+1] << endl;
      ASSERT(z % 2 == 0);
      const Player &p1 = pl[pair[z]];
      const Player &p2 = pl[pair[z+1]];
      ASSERT(p1.rank < p2.rank);  // p2 is lower half
      const float s1 = p1.score;
      const float s2 = p2.score;
      const int r1 = (IS_UNRATED(p1) ? unratedRating : p1.rating);
      const int r2 = (IS_UNRATED(p2) ? unratedRating : p2.rating);
      const int d2 = (sy == sx && s1 == s2 ? Min(r2 - rx, ry - r1) : r2 - rx);  // rule 29E5c
      //const int k1 = p1.rank;
      const int k2 = p2.rank;
      const bool u1 = (false && IS_UNRATED(p1) && threshold != 0);
      const bool u2 = (false && IS_UNRATED(p2) && threshold != 0);
      // only consider same score group and lower ranked (and not unrated) of the two
      // also check that pull-up is highest rated of anything in the score group (must also check higher ranked of each pair)
      //if (rx == 1611 && r2 == 1637)
        //cout << " (z=" << z << " s1=" << s1 << " s2=" << s2 << " r1=" << r1 << " r2=" << r2 << " d2=" << d2 << " k2=" << k2 << " u1=" << u1 << " u2=" << u2 << ")" << BR << endl;
      if (s1 == sx && !u1 && dl < r1 - rx		// check same score group for rated players with bigger transpose
		&& (sx < sy				// sx is pullup (check both upper and lower half); don't check whether k1 (upper half) is lower ranked;
			|| s1 > s2			// s1 is dropdown (must check upper half)
			|| p2.play_id == BYE_ID)) {	// include upper half if lower half is a bye (rule 28L2)
        cv += players * MAX_RATING + r1 - rx;
        //cout << "px: " << px << BR << endl;
        //cout << "py: " << py << BR << endl;
        //cout << "p1: " << p1 << BR << endl;
        //cout << "p2: " << p2 << BR << endl;
        //cout << " (z=" << z << " s1=" << s1 << " s2=" << s2 << " r1=" << r1 << " r2=" << r2 << " d2=" << d2 << " k2=" << k2 << " u1=" << u1 << " u2=" << u2 << ")" << BR << endl;
        //cout << "cv=" << cv/(players*MAX_RATING) << ',' << cv%(players*MAX_RATING) << BR << endl;
      }
      if (s2 == sx && !u2 && dl < d2			// check same score group for rated players with bigger transpose
		&& p2.play_id != BYE_ID			// but don't check p2's rating/ranking if it's a bye
		&& (sx < sy || k2 < kx)) {		// sx is pullup (check both) OR k2 (lower half) is lower ranked
        cv += players * MAX_RATING + d2;
        //cout << "px: " << px << BR << endl;
        //cout << "py: " << py << BR << endl;
        //cout << "p1: " << p1 << BR << endl;
        //cout << "p2: " << p2 << BR << endl;
        //cout << " (z=" << z << " s1=" << s1 << " s2=" << s2 << " r1=" << r1 << " r2=" << r2 << " d2=" << d2 << " k2=" << k2 << " u1=" << u1 << " u2=" << u2 << ")" << BR << endl;
        //cout << "cv=" << cv/(players*MAX_RATING) << ',' << cv%(players*MAX_RATING) << BR << endl;
      }
    }
#undef IS_UNRATED
#ifdef OLD_CODE
    if (dl < rm - rx)
      cv = players * MAX_RATING + rm - rx;
    else
      cv = 0;
    //cout << " rm=" << rm << endl;
#endif /* OLD_CODE */
  }
  if (cv != 0) {
    CostDescription(px.warn_codes, wCode, (
		threshold >= 200 ? "Transpose above 200 (29C1,29E5b,g)" :
		threshold >= 80 ? "Transpose above 80 (29C1,29E5b,g)" :
		"Transpose above 0 (29C1)"));
  }
  return cv;
}


// if even number, take lower of two in the middle
size_t MedianRating (const PlayerVector &pl, const IndexVector &pair, real score, size_t pBegin, size_t pEnd)
{
  ASSERT(pBegin % 2 == 0 && pEnd % 2 == 0);
  ASSERT(0 <= pBegin && pBegin < pEnd && pEnd <= pair.size());
  smallintVector sg1, sg2;
  for (size_t x = pBegin; x < pEnd; x += 2) {
    const Player &px = pl[pair[x]];
    const Player &py = pl[pair[x+1]];
    if (px.score == score && py.score == score && px.play_id != BYE_ID && py.play_id != BYE_ID) {
      sg1.push_back(px.rating);
      sg1.push_back(py.rating);
    }
    if (px.play_id != BYE_ID && !px.bye_request)
      sg2.push_back(px.rating);
    if (py.play_id != BYE_ID && !py.bye_request)
      sg2.push_back(py.rating);
  }
  if (sg1.size() > 0) {
    sort(sg1.begin(), sg1.end());
    if (sg1.size() % 2 == 1) return sg1[sg1.size()/2];
    return Min(sg1[sg1.size()/2], sg1[sg1.size()/2-1]);
  }
  if (sg2.size() > 0) {
    sort(sg2.begin(), sg2.end());
    if (sg2.size() % 2 == 1) return sg2[sg2.size()/2];
    return Min(sg2[sg2.size()/2], sg2[sg2.size()/2-1]);
  }
  return 0;
}

size_t UnratedRating (const PlayerVector &pl, const IndexVector &pair, real score, size_t pBegin, size_t pEnd)
{
  ASSERT(pBegin % 2 == 0 && pEnd % 2 == 0);
  ASSERT(0 <= pBegin && pBegin < pEnd && pEnd <= pair.size());
  smallint rating = MAX_RATING;
  for (size_t x = pBegin; x < pEnd; ++x) {
    const Player &px = pl[pair[x]];
    if (px.play_id != BYE_ID && !px.bye_request && px.score == score && px.rating < rating && (!px.is_unrated || px.use_rating == "none"))
      rating = px.rating;
  }
  return (rating == MAX_RATING ? 0 : rating);
}

size_t HighestRating (const PlayerVector &pl, const IndexVector &pair, real score, size_t pBegin, size_t pEnd)
{
  ASSERT(pBegin % 2 == 0 && pEnd % 2 == 0);
  ASSERT(0 <= pBegin && pBegin < pEnd && pEnd <= pair.size());
  smallint rating = 0;
  for (size_t x = pBegin; x < pEnd; ++x) {
    const Player &px = pl[pair[x]];
    if (px.play_id != BYE_ID && !px.bye_request && px.score == score && px.rating > rating)
      rating = px.rating;
  }
  return rating;
}

size_t PairingCard (char wCode, PlayerVector &pl, const IndexVector &pair, IndexSet &costPlayers)
{
#ifdef OLD_CODE
  if (pl[0].use_rating == "none")
    return 0;
#endif /* OLD_CODE */
  size_t num = 0;
#define COMPLETE	1
#define SMOOTH	1
  const string costDesc = "Transposed/Interchanged pair number (28A,28B,29A)";
  for (size_t x = 0; x < pair.size(); x += 2) {
    for (size_t y = x + 2; y < pair.size() && (COMPLETE || y <= x + 2); y += 2) {
      ASSERT(COMPLETE || y == x + 2);
      // transpose upper half
      if (pl[pair[x]].paired == pl[pair[y]].paired
		&& pl[pair[x]].score == pl[pair[y]].score
		&& (pl[pair[x]].rating == pl[pair[y]].rating || pl[pair[x]].rating == 0)
		&& pl[pair[x]].play_id != BYE_ID && pl[pair[y]].play_id != BYE_ID
		&& pl[pair[x]].rand > pl[pair[y]].rand) {
        if (SMOOTH)
          num += labs(pair[x] - pair[y]);
        else
          ++num;
        CostDescription(pl[pair[x]].warn_codes, wCode, costDesc.c_str());
        //static bool init = false; if (!init) { init = true; cout << "upper half: " << pl[pair[x]] << BR << pl[pair[y]] << BR << endl; }
        costPlayers.insert(pair[x]);
        costPlayers.insert(pair[y]);
      }
      // transpose lower half
      if (pl[pair[x+1]].paired == pl[pair[y+1]].paired
		&& pl[pair[x+1]].score == pl[pair[y+1]].score
		&& (pl[pair[x+1]].rating == pl[pair[y+1]].rating || pl[pair[x+1]].rating == 0)
		&& pl[pair[x+1]].play_id != BYE_ID && pl[pair[y+1]].play_id != BYE_ID
		&& pl[pair[x+1]].rand > pl[pair[y+1]].rand) {
        if (SMOOTH)
          num += labs(pair[x+1] - pair[y+1]);
        else
          ++num;
        CostDescription(pl[pair[x+1]].warn_codes, wCode, costDesc.c_str());
        //static bool init = false; if (!init) { init = true; cout << "lower half: " << pl[pair[x]] << BR << pl[pair[y]] << BR << endl; }
        costPlayers.insert(pair[x+1]);
        costPlayers.insert(pair[y+1]);
      }
    }
    ASSERT(x+1 < pair.size());
    ASSERT(pl[pair[x]].score >= pl[pair[x+1]].score);
    const bool isDropDown = (pl[pair[x]].score != pl[pair[x+1]].score || pl[pair[x+1]].play_id == BYE_ID);
    // interchange
    if (!isDropDown
	&& pl[pair[x]].paired == pl[pair[1]].paired
	&& pl[pair[x]].score == pl[pair[1]].score
	&& pl[pair[x]].rating == pl[pair[1]].rating
	&& (pl[pair[x]].rating == pl[pair[1]].rating || pl[pair[1]].rating == 0)
	&& pl[pair[x]].play_id != BYE_ID && pl[pair[1]].play_id != BYE_ID
	&& pl[pair[x]].rand > pl[pair[1]].rand) {
      if (SMOOTH)
        num += labs(pair[x] - pair[1]);
      else
        ++num;
      CostDescription(pl[pair[x]].warn_codes, wCode, costDesc.c_str());
      //static bool init = false; if (!init) { init = true; cout << "interchange: " << pl[pair[x]] << BR << endl; }
      costPlayers.insert(pair[x]);
      costPlayers.insert(pair[1]);
    }
    // dropdown
    if (isDropDown && x > 0
	&& pl[pair[x]].paired == pl[pair[x-1]].paired
	&& pl[pair[x]].score == pl[pair[x-1]].score
	&& pl[pair[x]].rating == pl[pair[x-1]].rating
	&& (pl[pair[x]].rating == pl[pair[x-1]].rating || pl[pair[x-1]].rating == 0)
	&& pl[pair[x]].play_id != BYE_ID && pl[pair[x-1]].play_id != BYE_ID
	&& pl[pair[x]].rand < pl[pair[x-1]].rand) {
      if (SMOOTH)
        num += labs(pair[x] - pair[x-1]);
      else
        ++num;
      CostDescription(pl[pair[x]].warn_codes, wCode, costDesc.c_str());
      //static bool init = false; if (!init) { init = true; cout << "dropdown: " << pl[pair[x]] << BR << endl; }
      costPlayers.insert(pair[x]);
      costPlayers.insert(pair[x-1]);
    }
  }
  return num;
}

CostValue ReversedColors (char wCode, Player &x, const Player &y, char xColor)
{
  const CostValue cv = x.board_color != xColor && xColor == 'W';
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Colors reversed for pair (28J;29E2,4)");
  return cv;
}

CostValue BoardOverlap (char wCode, const PlayerVector &pl, const IndexVector &pair, Player &x, const Player &y)
{
  CostValue cv = 0;
  if (x.rank < y.rank) {
    for (size_t z = 0; z < pair.size(); z += 2) {
      if (pl[pair[z+1]].play_id == BYE_ID) continue;
      //ASSERT(pl[pair[z]].board_num == pl[pair[z+1]].board_num);
      if ((x.play_id == pl[pair[z]].play_id && x.reentry == pl[pair[z]].reentry)
	  || (x.play_id == pl[pair[z+1]].play_id && x.reentry == pl[pair[z+1]].reentry))
        continue;
      if (x.board_num == pl[pair[z]].board_num)
        ++cv;
    }
  }
  if (cv != 0) CostDescription(x.warn_codes, wCode, "Board number overlap (28J)");
  return cv;
}

CostValue BoardOrder (char wCode, const PlayerVector &pl, const IndexVector &pair, Player &px, Player &py, size_t x, size_t y, size_t pBegin, size_t pEnd)
{
  CostValue cv = 0;
  ASSERT(abs(int(x-y)) == 1);
  const size_t w = Min(x,y);
  if (px < py && px.play_id != BYE_ID && py.play_id != BYE_ID && pBegin+2 <= w && w < pEnd) {
    //if (px.board_num != py.board_num)
      //cout << px << BR << py << BR << endl;
    //ASSERT(px.board_num == py.board_num);
    const Player &pz2 = pl[pair[w-2]];
    const Player &pz1 = pl[pair[w-1]];
    if (pz2.board_num > Min(px.board_num,py.board_num) && pz1.board_num > Min(px.board_num,py.board_num)
		&& pz1.paired == py.paired && pz2.paired == py.paired
		&& pz1.play_id != BYE_ID && pz2.play_id != BYE_ID
		) {
      //if (pz2.board_num != pz1.board_num)
        //cout << pz2 << BR << pz1 << BR << endl;
      //ASSERT(pz2.board_num == pz1.board_num);
      ++cv;
    }
  }
  if (cv != 0) CostDescription(py.warn_codes, wCode, "Board number order (28J)");
  return cv;
}

#if PERF_DEBUG
static uint64_t costCount = 0;
static vector<uint64_t> sTry(8,0);
static vector<uint64_t> sDo(8,0);
#endif

Cost CostFunction (PlayerVector &pl, const IndexVector &pair, size_t remainingRounds, size_t pBegin, size_t pEnd, bool doCodes, const bool usePairableCost, IndexSet &costPlayers)
{
#if DEBUG
  cout << "CostFunction(" << pl.size() << ',' << pair.size() << ',' << remainingRounds << ',' << pBegin << ',' << pEnd << ',' << doCodes << ',' << usePairableCost << ")"BR << endl;
  //cout << pl << BR << endl;
#endif
#if PERF_DEBUG
  ++costCount;
#endif
  ASSERT(pair.size() % 2 == 0);
  ASSERT(pl.size() >= 1 && pl.back().play_id == BYE_ID);
  for (size_t x = 0; x < pl.size(); ++x) {
    //cout << x << ": " << pl[x] << BR << endl;
    //cout << "pl[" << x << "].opponent_ranks=" << pl[x].opponent_ranks << BR << endl;
    ASSERT(x == 0 || pl[x-1] < pl[x]);
  }
  ASSERT(pBegin % 2 == 0 && pEnd % 2 == 0);
  ASSERT(0 <= pBegin && pBegin <= pEnd && pEnd <= pair.size());
  while (pBegin < pEnd && pl[pair[pEnd-1]].play_id == BYE_ID && (pl[pair[pEnd-2]].bye_request || pl[pair[pEnd-2]].bye_house))
    pEnd -= 2;  // don't evaluate the granted bye requests
  Cost c;  // defaults to zero for each field
  real lastScore = -1;
  smallint lastMedian = 0;
  smallint lastUnrated = 0;
  smallint lastHighest = 0;
  c.players = pl.size() - 1;
  if (doCodes)
    for (size_t x = pBegin; x < pEnd; ++x)
      pl[pair[x]].warn_codes = string();
  char wCode = 'A' - 1;
  #define WCODE	(wCode == 'Z' ? wCode='a' : ++wCode)	/* increment should skip over non-letter characters */
#if USE_PAIRABLE_COST
  char wCodePlayers = 'A';
#if !USE_28N3_0
  char wCodeTeams = 'B';
#endif /* !USE_28N3_0 */
#endif /* USE_PAIRABLE_COST */
  char wCodePairCard = 'C';
  bool isHousePlayer = false;
  real lowestScore = (pl.size() <= 0 || pair.size() <= 0 ? 0 : pl[pair[0]].score);
  for (size_t x = pBegin; x < pEnd; x += 2) {
    const Player &px = pl[pair[x]];
    const Player &py = pl[pair[x+1]];
    if (lowestScore > px.score)
      lowestScore = px.score;
    if (lowestScore > py.score)
      lowestScore = py.score;
  }
  
  for (size_t x = pBegin; x < pEnd; x += 2) {
    const Cost lastC = c;
    wCode = 'A' - 1;
    Player &px = pl[pair[x]];  // not const so we can update warn_codes
    Player &py = pl[pair[x+1]];
    if (px.bye_house || py.bye_house)
      isHousePlayer = true;
    //cout << "x=" << x << " pair[x]=" << pair[x] << " px.rank=" << px.rank << " pair[x+1]=" << pair[x+1] << " py.rank=" << py.rank << BR << endl;
    const char xColor = AllocateColor(px, py, x/2%2==0);
    const smallint mx = (px.score == lastScore ? lastMedian : MedianRating(pl, pair, px.score, pBegin, pEnd));
    const smallint my = (py.score == lastScore ? lastMedian : py.score == px.score ? mx : MedianRating(pl, pair, py.score, pBegin, pEnd));
    const smallint ux = (px.score == lastScore ? lastUnrated : UnratedRating(pl, pair, px.score, pBegin, pEnd));
    const smallint uy = (py.score == lastScore ? lastUnrated : py.score == px.score ? ux : UnratedRating(pl, pair, py.score, pBegin, pEnd));
    const smallint hx = (px.score == lastScore ? lastHighest : HighestRating(pl, pair, px.score, pBegin, pEnd));
    const smallint hy = (py.score == lastScore ? lastHighest : py.score == px.score ? hx : HighestRating(pl, pair, py.score, pBegin, pEnd));
    //if (doCodes && (px.uscf_id == 15246688 || py.uscf_id == 15246688))
      //cout << px << BR << py << BR << "mx=" << mx << " my=" << my << " ux=" << ux << " uy=" << uy << BR << endl;
    if (lastScore != px.score) {
      lastScore = px.score;
      lastMedian = mx;
      lastUnrated = ux;
    }
    #define F2(f)		(WCODE, f(doCodes*wCode, px, py) + f(doCodes*wCode, py, px))
    #define F2_V1(f,v1)		(WCODE, f(doCodes*wCode, px, py, v1) + f(doCodes*wCode, py, px, v1))
    #define F2_V2(f,v1,v2)	(WCODE, f(doCodes*wCode, px, py, v1, v2) + f(doCodes*wCode, py, px, v1, v2))
    #define F2_PLAY(f)		F2_V1(f, pl.size())
    #define F2_RND(f)		F2_V1(f, remainingRounds)
    #define F2_SCORE(f)		F2_V1(f, lowestScore)
    #define F2_RND_SCORE(f)	F2_V2(f, remainingRounds, lowestScore)
    #define F2_COLOR(f)		(WCODE, f(doCodes*wCode, px, py, xColor) + f(doCodes*wCode, py, px, FlipColor(xColor)))
    #define F2_PLAY_COLOR(f)	(WCODE, f(doCodes*wCode, px, py, pl.size(), xColor) + f(doCodes*wCode, py, px, pl.size(), FlipColor(xColor)))
    #define F2_PLAY_RND(f)	F2_V2(f, pl.size(), remainingRounds)
    #define F2_PLAY_SCORE(f)	F2_V2(f, pl.size(), lowestScore)
    #define INTERCHANGE(num)	(WCODE, Interchange(doCodes*wCode, px, py, pl.size(), mx, hx, ux, num) + Interchange(doCodes*wCode, py, px, pl.size(), my, hy, uy, num))
    #define TRANSPOSE(num)	(WCODE, Transpose(doCodes*wCode, pl, pair, x, x+1, ux, num, pBegin, pEnd) + Transpose(doCodes*wCode, pl, pair, x+1, x, uy, num, pBegin, pEnd))
    #define BOARD_OVERLAP	(WCODE, BoardOverlap(doCodes*wCode, pl, pair, px, py) + BoardOverlap(doCodes*wCode, pl, pair, py, px))
    #define BOARD_ORDER		(WCODE, BoardOrder(doCodes*wCode, pl, pair, px, py, x, x+1, pBegin, pEnd) + BoardOrder(doCodes*wCode, pl, pair, py, px, x+1, x, pBegin, pEnd))
    c.byeChoice += F2(ByeChoice);
    c.byeAgain += F2_PLAY(ByeAgain);
    c.playersMeetTwice += F2_PLAY_COLOR(IdenticalMatch);
    c.playersMeetTwice += F2_PLAY(PlayersMeetTwice);
#if USE_PAIRABLE_COST
    wCodePlayers = WCODE;
#endif /* USE_PAIRABLE_COST */
#if !USE_28N3_0
    c.teamBlocks2 += F2_PLAY(TeamBlocks2);
#endif /* !USE_28N3_0 */
    c.unequalScores += F2_PLAY_RND(UnequalScores);
    c.teamBlocks += F2_PLAY(TeamBlocks);
    //if (c.teamBlocks != lastC.teamBlocks)
      //cout << "team block: " << pair[x] << ' ' << pair[x+1] << BR << endl;
#if USE_PAIRABLE_COST
#if !USE_28N3_0
    wCodeTeams = WCODE;
#endif /* !USE_28N3_0 */
#endif /* USE_PAIRABLE_COST */
    c.byeAfterHalf += F2_PLAY(ByeAfterHalf);
    c.lowestScoreBye += F2_PLAY_SCORE(LowestScoreBye);
    c.lowestRatedBye += F2_RND(LowestRatedBye);
    c.oddPlayerUnrated += F2(OddPlayerUnrated);
    c.oddPlayerMultipleGroups += F2_PLAY(OddPlayerMultipleGroups);
    c.interchange200 += INTERCHANGE(200);
    c.transpose200 += TRANSPOSE(200);
    if (px.multiround % 2 == 1) {
      c.colorImbalance += F2_COLOR(ColorImbalance);
      c.colorRepeat3 += F2_COLOR(ColorRepeat3);
    }
    c.interchange80 += INTERCHANGE(80);
    c.transpose80 += TRANSPOSE(80);
    if (px.multiround % 2 == 1)
      c.colorAlternate += F2_COLOR(ColorAlternate);
    c.interchange0 += INTERCHANGE(0);
    c.transpose0 += TRANSPOSE(0);
    wCodePairCard = WCODE;
    if (doCodes) {
      c.reversedColors += F2_COLOR(ReversedColors);
      c.boardOverlap += BOARD_OVERLAP;
      c.boardOrder += BOARD_ORDER;
    }
    #undef BOARD_ORDER
    #undef BOARD_OVERLAP
    #undef TRANSPOSE
    #undef INTERCHANGE
    #undef F2_PLAY_SCORE
    #undef F2_PLAY_RND
    #undef F2_COLOR
    #undef F2_RND_SCORE
    #undef F2_SCORE
    #undef F2_RND
    #undef F2_PLAY
    #undef F2_V2
    #undef F2_V1
    #undef F2
    ASSERT(('A' <= wCode && wCode <= 'Z') || ('a' <= wCode && wCode <= 'z'));
    if (c != lastC) {
      costPlayers.insert(pair[x]);
      if (x+1 < pEnd)
        costPlayers.insert(pair[x+1]);
    }
  }
  // must have at least one bye when odd number of players and no house player
  // removing this cost allows zero cost to end the search for optimal
  c.byeChoice -= (!isHousePlayer && pEnd > 0 && pl[pair[pEnd-1]].play_id == BYE_ID && !pl[pair[pEnd-2]].bye_request);
  //cout << "calling PairableCost()"BR << endl;
  //if (pl.size() <= pl[0].rnd + remainingRounds + 10) {
#if USE_PAIRABLE_COST
  if (usePairableCost) {
    c.cantPairPlayers = PairableCost(doCodes*wCodePlayers, pl, pair, remainingRounds, false);
#if !USE_28N3_0
    if (!c.cantPairPlayers)
      c.cantPairTeams = PairableCost(doCodes*wCodeTeams, pl, pair, remainingRounds, true);
#endif /* !USE_28N3_0 */
  }
#endif /* USE_PAIRABLE_COST */
  c.pairingCard = PairingCard(doCodes*wCodePairCard, pl, pair, costPlayers);
  if (doCodes)
    for (size_t x = 0; x < pl.size(); ++x)
      sort(pl[x].warn_codes.begin(), pl[x].warn_codes.end());
#if DEBUG
  cout << "CostFunction() done."BR << endl;
#endif
  return c;
}

Cost CostFunction (PlayerVector &pl, const IndexVector &pair, size_t remainingRounds, size_t pBegin, size_t pEnd, bool doCodes, const bool usePairableCost)
{
  IndexSet costPlayers;
  return CostFunction(pl, pair, remainingRounds, pBegin, pEnd, doCodes, usePairableCost, costPlayers);
}

////////////////////////  OTHER PROCEDURES  ////////////////////////

/*
PlayerVector contains list of all non-bye players (not just ones to be paired)
	implementation appends one bye and sorts (bye is deleted when done pairing):
		bye at end
		not bye requests before bye requests
		not paired before paired
		descending ranks according to USCF rules
	in this way, players that want pairings are at the beginning of the array
		and we have all the players to enable checks for future rounds
IndexVector contains indices to players (and byes) sorted by board (i.e pairing)
	board position is array position divided by two (two players on same board)
	first is higher ranked; second is lower ranked
*/

void AssertNoDuplicates (const PlayerVector &pl, const IndexVector &pair)
{
  for (size_t x = 0; x < pair.size(); ++x) {
    for (size_t y = x+1; y < pair.size(); ++y) {
      if (pl[pair[x]].play_id == pl[pair[y]].play_id && pl[pair[x]].play_id != BYE_ID)
        cout << BR"x=" << x << " y=" << y << BR << pl[pair[x]] << BR << pl[pair[y]] << BR << endl;
      ASSERT(pl[pair[x]].play_id != pl[pair[y]].play_id || pl[pair[x]].play_id == BYE_ID);
    }
  }
}

// insertion sort to put players on correct boards
// active but not paired comes first
void SortBoards (const PlayerVector &pl, IndexVector &pair)
{
  //cout << "SortBoards(" << pl.size() << ',' << pair.size() << ")"BR << endl;
  //cout << "pair=";
  //for (size_t x = 0; x < pair.size(); ++x)
    //cout << (x%2==0?" W:":" B:") << pl[pair[x]].play_id << '_' << pl[pair[x]].reentry;
  //cout << BR << endl;
  ASSERT(pl.back().play_id == BYE_ID);
  ASSERT(pair.size() % 2 == 0);
  for (size_t x = 0; x < pair.size(); x += 2) {
    //cout << " x=" << x << flush;
    ASSERT(0 <= pair[x] && pair[x] < pl.size());
    ASSERT((0 <= pair[x+1] && pair[x+1] < pl.size()));
    for (size_t y = x; y > 0; y -= 2) {
      //cout << " y=" << y << flush;
      //cout << " pair[y-2]=" << pair[y-2] << " pair[y-1]=" << pair[y-1] << " pair[y]=" << pair[y] << " pair[y+1]=" << pair[y+1] << BR << endl;
      ASSERT(pair[y+1] != pair[y] /*&& pair[y+1] != pair[y-1]*/ && pair[y+1] != pair[y-2]
		&& pair[y] != pair[y-1] && pair[y] != pair[y-2]
		&& pair[y-1] != pair[y-2]);
      ASSERT(pair[y+1] != pair[y-1] || (pl[pair[y+1]].play_id == BYE_ID && pl[pair[y-1]].play_id == BYE_ID));

      if (pl[pair[y-2]].paired < pl[pair[y]].paired || (pl[pair[y-2]].paired == pl[pair[y]].paired
		&& ((pl[pair[y-1]].play_id == BYE_ID) < (pl[pair[y+1]].play_id == BYE_ID)
			|| ((pl[pair[y-1]].play_id == BYE_ID) == (pl[pair[y+1]].play_id == BYE_ID)
		/* if same rank for top players, then look at bottom players before using pairing number */
		&& (pl[pair[y-2]].bye_request < pl[pair[y]].bye_request || (pl[pair[y-2]].bye_request == pl[pair[y]].bye_request
		&& (pl[pair[y-2]].score > pl[pair[y]].score || (pl[pair[y-2]].score == pl[pair[y]].score
		&& (pl[pair[y-1]].score > pl[pair[y+1]].score || (pl[pair[y-1]].score == pl[pair[y+1]].score
		&& (pl[pair[y-2]].rating > pl[pair[y]].rating || (pl[pair[y-2]].rating == pl[pair[y]].rating
		&& (pl[pair[y-1]].rating > pl[pair[y+1]].rating || (pl[pair[y-1]].rating == pl[pair[y+1]].rating
		&& pl[pair[y-2]] <= pl[pair[y]]))))))))))))))
	break;
      ASSERT(pl[pair[y]].play_id != BYE_ID && pl[pair[y-2]].play_id != BYE_ID);
#ifdef OLD_CODE
      ASSERT((pl[pair[y+1]].play_id != BYE_ID && pl[pair[y-1]].play_id != BYE_ID)
		|| (pl[pair[y+1]].play_id == BYE_ID && pl[pair[y-1]].play_id == BYE_ID));
#endif /* OLD_CODE */
      swap(pair[y], pair[y-2]);
      swap(pair[y+1], pair[y-1]);
    }
  }
  //cout << " SortBoards() done."BR << endl;
  //cout << "pair=";
  //for (size_t x = 0; x < pair.size(); ++x)
    //cout << (x%2==0?" W:":" B:") << pl[pair[x]].play_id << '_' << pl[pair[x]].reentry;
  //cout << BR << endl;
}

// setup initial position (the given hint) for pairing search
void HintPairings (const PlayerVector &pl, IndexVector &pair, bool collapseByes)
{
#if DEBUG
  cout << "HintPairings(" << pl.size() << ',' << pair.size() << ',' << collapseByes << ")"BR << endl;
  AssertNoDuplicates(pl, pair);  // pair.size() is typically zero, in which case this does nothing
#endif
  ASSERT(pl.size() > 0 && pl.back().play_id == BYE_ID);
  for (size_t x = 0; x < pl.size(); ++x) {
    ASSERT(x == 0 || pl[x-1] < pl[x]);
    ASSERT(pl[x].rank == integer(x));
  }
  //cout << "board_num=";
  //for (size_t x = 0; x < pl.size(); ++x)
    //cout << ' ' << pl[x].board_num << '('
	//<< pl[x].play_id << '_' << pl[x].reentry
	//<< " bye=" << pl[x].bye_request << " paired=" << pl[x].paired
	//<< ')';
  //cout << BR << endl;
  typedef std::pair<const integer, size_t> Elem;
  typedef std::multimap<integer, size_t> Map;
  typedef Map::const_iterator CIter;
  Map m;
  for (size_t x = 0; x < pl.size()-1; ++x)
    if (pl[x].board_num != -1)
      m.insert(Elem(pl[x].board_num, x));
  //cout << "map container: ";
  //for (CIter i = m.begin(); i != m.end(); ++i)
    //cout << '(' << i->first << ',' << i->second << ')';
  //cout << BR << endl;

  pair.clear();		// preserved pairings
  IndexVector single	// orphans that need pairing
	, other;	// non-paired players
  const size_t byeIndex = pl.size()-1;
  for (CIter i = m.begin(); i != m.end(); ++i) {
    const Elem e1 = *i;
    const Player &p1 = pl[e1.second];
    //cout << "e1.first=" << e1.first << " e1.second=" << e1.second << " p1=" << p1.play_id << '_' << p1.reentry << BR << endl;
    CIter j = i;
    ++j;
    if (j == m.end()) {
      // last board originally scheduled for a bye
      if (p1.paired || p1.bye_request || !collapseByes) {
        other.push_back(p1.rank);
        other.push_back(byeIndex);
      } else {
        single.push_back(p1.rank);
      }
    } else {
      const Elem e2 = *j;
      const Player &p2 = pl[e2.second];
      //cout << "e2.first=" << e2.first << " e2.second=" << e2.second << " p2=" << p2.play_id << '_' << p2.reentry << BR << endl;
      if (p2.board_num != p1.board_num || p2.paired != p1.paired || (!p1.paired && (p1.bye_request || p2.bye_request))) {
        // service only p1, leaving p2 for next iteration
        //cout << " service p1 only"BR << endl;
        if (p1.paired || p1.bye_request || !collapseByes) {
          other.push_back(p1.rank);
          other.push_back(byeIndex);
        } else {
          single.push_back(p1.rank);
        }
      } else {
        // service p1 and p2
        //cout << " service p1 and p2"BR << endl;
        if (p1.paired) {
          other.push_back(p1.rank);
          other.push_back(p2.rank);
        } else {
          pair.push_back(p1.rank);
          pair.push_back(p2.rank);
        }
        ++i;
        ASSERT(i == j);
      }
    }
  }

  // merge arrays
  //cout << "merging before:" << endl;
  //cout << " pair: " << pair << endl;
  //cout << " single: " << single << endl;
  //cout << " other: " << other << BR << endl;
  //cout << "pair.size()=" << pair.size() << " other.size()=" << other.size() << " collapseByes=" << collapseByes << " single.size()=" << single.size() << BR << endl;
  pair.insert(pair.end(), single.begin(), single.end());
  if (pair.size() % 2 != 0)
    pair.insert(pair.end(), byeIndex);
  pair.insert(pair.end(), other.begin(), other.end());
  //cout << "merging after:" << endl;
  //cout << " pair: " << pair << BR << endl;

  // put players on correct boards
  for (size_t x = 0; x < pair.size(); x += 2)
    if (pl[pair[x]].rank > pl[pair[x+1]].rank)
      swap(pair[x], pair[x+1]);
  SortBoards(pl, pair);
#if DEBUG
  cout << "done HintPairings()"BR << endl;
  AssertNoDuplicates(pl, pair);
#endif
}

void ColorLookahead (PlayerVector &pl, IndexVector &pair, size_t players, smallint totalRounds, const sizeVector &num, const vector<sizeVector> &color)
{
  bool isX = true;
  for (size_t x = 0; x < color.size(); ++x)
    if (num[x] != color[x][2])
      isX = false;
  if (isX) return;  // nothing to change
}

// determine pairings for each score group (rule 27A2) without regard to prior opponents, teammates, or color 
// this will be the correct pairings for the first round if there are no team blocks
// this will be close to the correct pairings (except for due color) for large number of players with few team blocks
void FirstPairings (PlayerVector &pl, IndexVector &pair, size_t players, smallint totalRounds)
{
  const bool isLookahead = true;
#if DEBUG
  cout << "FirstPairings(" << pl.size() << ',' << pair.size() << ',' << players << ")"BR << endl;
  AssertNoDuplicates(pl, pair);
#endif
  ASSERT(players <= pair.size());
  for (size_t x = 0; x < players; ++x)
    ASSERT(pl[pair[x]].play_id != BYE_ID);
  //cout << "unsorted pair=";
  //for (size_t x = 0; x < pair.size(); ++x)
    //cout << (x%2==0?" W:":" B:") << pl[pair[x]].play_id << '_' << pl[pair[x]].reentry;
  //cout << BR << endl;

  // push byes to the end (also naive 1 vs 2 pairings)
  sort(pair.begin(), pair.end() - (pair.size()-players));
  //cout << "sorted pair=";
  //for (size_t x = 0; x < pair.size(); ++x)
    //cout << (x%2==0?" W:":" B:") << pl[pair[x]].play_id << '_' << pl[pair[x]].reentry;
  //cout << BR << endl;

  // for each score group
  //cout << "before each score group"BR << endl;
  AssertNoDuplicates(pl, pair);
  ASSERT(players % 2 == 0 || (players < pair.size() && pl[pair[players]].play_id == BYE_ID));
  sizeVector num((pl.size() == 0 ? 0 : 2 * pl[0].score + 1), 0);
  vector<sizeVector> color((pl.size() == 0 ? 0 : 2 * pl[0].score + 1), sizeVector(3,0));
  for (size_t x = 0; x < players; ) {
    //cout << " x=" << x << endl;
    AssertNoDuplicates(pl, pair);
    ASSERT(x % 2 == 0);
    const real scoreGroup = pl[x].score;
    // find end of score group
    for (size_t y = x + 1; ; ++y) {
      //cout << " y=" << y << endl;
      AssertNoDuplicates(pl, pair);
      ++num[2*scoreGroup];
      ++color[2*scoreGroup][toupper(pl[y-1].due_color[0]) == 'W' ? 0 : toupper(pl[y-1].due_color[0]) == 'B' ? 1 : 2];
      // if end of score group
      if (y >= players || pl[y].score != scoreGroup) {
        ASSERT(num[2*scoreGroup] == y - x);
        //cout << " num=" << num[2*scoreGroup] << endl;
        // for each board
        for (size_t z = 0; z < num[2*scoreGroup]-1; z += 2) {
          // assign upper half against lower half
          //cout << " z=" << z << endl;
          const size_t upper = x+z/2;
          const size_t lower = x+num[2*scoreGroup]/2+z/2;
          //cout << " upper(" << x+z << ")=" << upper << " lower(" << x+z+1 << ")=" << lower << endl;
          pair[x+z] = upper;  // upper half in score group
          pair[x+z+1] = lower;  // lower half in score group
          //cout
		//<< " pair[" << x+z << "]=" << pair[x+z] << " play_id=" << pl[pair[x+z]].play_id
		//<< " pair[" << x+z+1 << "]=" << pair[x+z+1] << " play_id=" << pl[pair[x+z+1]].play_id
		//<< endl;
        }
        //cout << " after z" << endl;
        AssertNoDuplicates(pl, pair);
        // handle potential odd player
        if (num[2*scoreGroup] % 2 == 0) {
          //cout << " no odd player"BR << endl;
          x = y;  // no odd player
        } else if (y < players) {
          //cout << " odd player drop"BR << endl;
          pair[y-1] = y-1;  // odd player drop down
          pair[y] = y;  // highest rated in next score group
          x = y + 1;
        } else {
          //cout << " odd player bye"BR << endl;
          pair[y-1] = y-1;  // odd player bye
          ASSERT(y < pl.size());
          //cout << "pair=";
          //for (size_t w = 0; w < pair.size(); ++w)
            //cout << (w%2==0?" W:":" B:") << pl[pair[w]].play_id << '_' << pl[pair[w]].reentry;
          //cout << BR << endl;
          //cout << "pair["<<y<<"]=" << pair[y] << BR << endl;
          //cout << "pl[pair["<<y<<"]]=" << pl[pair[y]] << BR << endl;
          ASSERT(pl[pair[y]].play_id == BYE_ID);
          x = y;
        }
        break;  // next x
      }
    }
  }
  if (isLookahead)
    ColorLookahead(pl, pair, players, totalRounds, num, color);
#if DEBUG
  cout << "done FirstPairings()"BR << endl;
  AssertNoDuplicates(pl, pair);
#endif
}

void RotatePairDown (IndexVector &pair, size_t x, size_t y, size_t pBegin, size_t pEnd, bool oddDropDown, bool oddPullUp, const BoolVector &shift)
{
  //cout << "RotatePairDown(" << pair.size() << ',' << x << ',' << y << ',' << pBegin << ',' << pEnd << ',' << oddDropDown << ',' << oddPullUp << ")"BR << endl;
  ASSERT(pBegin % 2 == 0 && pBegin <= x && x < y && y <= pEnd && pEnd % 2 == 0);
  if (oddDropDown) {
    ASSERT(y % 2 == 0 && y == pEnd - 2);
    --y;
    pEnd -= 2;
  }
  if (oddPullUp) {
    ASSERT(x % 2 == 1 && x == pBegin + 1);
    ++x;
    pBegin += 2;
    swap(pair[x-1], pair[x]);
  }
  ASSERT(pBegin % 2 == 0 && pBegin <= x && x <= y && y <= pEnd && pEnd % 2 == 0);
  if (x % 2 == 0) {
    if (y % 2 == 0) {
      for (size_t z = x; z+2 <= y; z += 2)
        swap(pair[z+shift[z]], pair[z+2+shift[z+2]]);
    } else {
      for (size_t z = x; z+2 < pEnd; z += 2)
        swap(pair[z+shift[z]], pair[z+2+shift[z+2]]);
      swap(pair[pEnd-2], pair[pBegin+1]);
      for (size_t z = pBegin+1; z+2 <= y; z += 2)
        swap(pair[z+shift[z]], pair[z+2+shift[z+2]]);
    }
  } else {
    if (y % 2 == 0) {
      for (size_t z = y; z+2 < pEnd; z += 2)
        swap(pair[z+shift[z]], pair[z+2+shift[z+2]]);
      swap(pair[pEnd-2], pair[pBegin+1]);
      for (size_t z = pBegin+1; z+2 <= x; z += 2)
        swap(pair[z+shift[z]], pair[z+2+shift[z+2]]);
    } else {
      for (size_t z = x; z+2 <= y; z += 2)
        swap(pair[z+shift[z]], pair[z+2+shift[z+2]]);
    }
  }
  if (oddDropDown)
    swap(pair[y], pair[y+1]);
  //cout << "done RotatePairDown()"BR << endl;
}

void RotatePairUp (IndexVector &pair, size_t x, size_t y, size_t pBegin, size_t pEnd, bool oddDropDown, bool oddPullUp, const BoolVector &shift)
{
  //cout << "RotatePairUp(" << pair.size() << ',' << x << ',' << y << ',' << pBegin << ',' << pEnd << ',' << oddDropDown << ',' << oddPullUp << ")"BR << endl;
  ASSERT(pBegin % 2 == 0 && pBegin <= x && x < y && y <= pEnd && pEnd % 2 == 0);
  if (oddDropDown) {
    ASSERT(y % 2 == 0 && y == pEnd - 2);
    --y;
    pEnd -= 2;
    swap(pair[y+1], pair[y]);
  }
  if (oddPullUp) {
    ASSERT(x % 2 == 1 && x == pBegin + 1);
    ++x;
    pBegin += 2;
  }
  ASSERT(pBegin % 2 == 0 && pBegin <= x && x <= y && y <= pEnd && pEnd % 2 == 0);
  if (x % 2 == 0) {
    if (y % 2 == 0) {
      for (size_t z = y; z >= x+2; z -= 2)
        swap(pair[z+shift[z]], pair[z-2+shift[z-2]]);
    } else {
      for (size_t z = y; z >= pBegin+2; z -= 2)
        swap(pair[z+shift[z]], pair[z-2+shift[z-2]]);
      swap(pair[pBegin+1], pair[pEnd-2]);
      for (size_t z = pEnd-2; z >= x+2; z -= 2)
        swap(pair[z+shift[z]], pair[z-2+shift[z-2]]);
    }
  } else {
    if (y % 2 == 0) {
      for (size_t z = x; z >= pBegin+2; z -= 2)
        swap(pair[z+shift[z]], pair[z-2+shift[z-2]]);
      swap(pair[pBegin+1], pair[pEnd-2]);
      for (size_t z = pEnd-2; z >= y+2; z -= 2)
        swap(pair[z+shift[z]], pair[z-2+shift[z-2]]);
    } else {
      for (size_t z = y; z >= x+2; z -= 2)
        swap(pair[z+shift[z]], pair[z-2+shift[z-2]]);
    }
  }
  if (oddPullUp)
    swap(pair[x], pair[x-1]);
  //cout << "done RotatePairUp()"BR << endl;
}

// returns true if changed
bool RotateColor (const PlayerVector &pl, IndexVector &pair, size_t x, size_t y, size_t pBegin, size_t pEnd, bool oddDropDown, bool oddPullUp)
{
  if (x/2+1 >= y/2) return false;  // at least one row separating ... otherwise, simple swap would be sufficient
  const Player &px = pl[pair[x]], &py = pl[pair[y]];
  if (px.score != py.score) return false;  // must be same score
  char xColor = toupper(px.due_color[0] == 'x' ? FlipColor(py.due_color[0]) : px.due_color[0]);
  char yColor = toupper(py.due_color[0] == 'x' ? FlipColor(px.due_color[0]) : py.due_color[0]);
  if (xColor == yColor) return false;  // must be different colors
  ASSERT(xColor != 'X' && yColor != 'X');
  const bool isFlipX = (xColor == toupper(px.due_color[0]) && yColor == toupper(py.due_color[0]));
  #define OPP(v)	pl[pair[(v)+((v)%2==0?1:-1)]]
  #define COLOR(v)	toupper(pl[pair[v]].due_color[0] != 'x' ? pl[pair[v]].due_color[0] : OPP(v).due_color[0] == 'x' ? ((v)%2==0?'W':'B') : isFlipX ? OPP(v).due_color[0] : FlipColor(OPP(v).due_color[0]))
  size_t top = x;
  if (oddPullUp || x % 2 == 0) {
    ASSERT(!oddPullUp || OPP(x).score > px.score);
    for (top = x/2*2+2; top < y/2*2 && COLOR(top) == xColor; top += 2)
      ;  // find color change
    if (top >= y/2*2)
      return false;  // not enough color changes (need one more)
    for (size_t z = top; ; z -= 2) {
      if (z == x || z+1 == x) {
        swap(pair[x], pair[z+2]);
        ++top;
        break;
      }
      swap(pair[z], pair[z+2]);
    }
  }
  ASSERT(top % 2 == 1);

  if (oddDropDown || y % 2 == 0) {
#ifdef NEW_CODE
    // TBD: fix this
    if (oddDropDown && py.score <= OPP(y).score) {
      cout << "pl=" << pl << BR << endl;
      cout << "pair=" << pair << BR << endl;
      cout << "x=" << x << " y=" << y << " pBegin=" << pBegin << " pEnd=" << pEnd << " oddDropDown=" << oddDropDown << " oddPullUp=" << oddPullUp << BR << endl;
      cout << "oddDropDown=" << oddDropDown << " py.score=" << py.score << " OPP(y).score=" << OPP(y).score << BR << endl;
    }
    ASSERT(!oddDropDown || py.score > OPP(y).score);
#endif /* NEW_CODE */
    size_t w = top;
    for (size_t z = w + 2; z < y; z += 2) {
      ASSERT(pBegin <= z-2 && z-2 <= pEnd);
      if (COLOR(z) == yColor) {
        swap(pair[w], pair[z]);
        w = z;
      }
    }
    swap(pair[w], pair[y]);
    w = y;
    for (size_t z = w + 1; z > top+2; z -= 2) {
      ASSERT(pBegin <= z && z <= pEnd);
      if (COLOR(z-2) == xColor) {
        swap(pair[w], pair[z-2]);
        w = z-2;
      }
    }
  } else {
    for (size_t z = top; z >= x+4; z -= 2) {
      ASSERT(pBegin <= z && z <= pEnd);
      swap(pair[z], pair[z-2]);
    }
    swap(pair[top], pair[y]);
  }
  return true;
}

// search for minimal-cost pairings (according to CostFunction) in global space of all possible pairings
// pBegin and pEnd are range of pair indices, not pair values
Cost MinimizePairingCost (PlayerVector &pl, IndexVector &pair, const size_t remainingRounds, const int depth, const size_t pBegin, const size_t pEndConst, const bool usePairableCost)
{
#if PERF_DEBUG
  cout << "Begin time: " << flush; system("date"); cout << " " << endl;
  costCount = 0;
  for (size_t x = 0; x < sTry.size(); ++x)
    sTry[x] = 0;
  for (size_t x = 0; x < sDo.size(); ++x)
    sDo[x] = 0;
#endif
#if DEBUG
  cout << "MinimizePairingCost(" << pl.size() << ',' << pair.size() << ',' << remainingRounds << ',' << depth << ',' << pBegin << ',' << pEndConst << ")"BR << endl;
  AssertNoDuplicates(pl, pair);
#endif
#if !USE_PAIRABLE_COST
  cout << "WARNING: PairableCost() feature turned off; no multi-round look-ahead used to avoid players meeting twice in small sections"BR << endl;
#endif /* USE_PAIRABLE_COST */
  //for (size_t x = 0; x < pl.size(); ++x)
    //cout << pl[x] << BR << endl;
  //cout << pair << BR << endl;
  size_t pEnd = pEndConst;
  const bool hasBye = (pEnd % 2 != 0);
  if (hasBye && pEnd < pair.size() && pl[pair[pEnd]].play_id == BYE_ID)
    ++pEnd;
  //cout << "pBegin=" << pBegin << " pEnd=" << pEnd << BR << endl;
  ASSERT(pBegin % 2 == 0 && pEnd % 2 == 0);
  ASSERT(0 <= pBegin && pBegin <= pEnd && pEnd <= pair.size());
  IndexVector bestPair = pair;
  //cout << "bestPair: " << bestPair << BR << endl;
  IndexSet bestCostPlayers;
  Cost bestCost = CostFunction(pl, bestPair, remainingRounds, pBegin, pEnd, false, usePairableCost, bestCostPlayers);
  //return bestCost;
  //cout << "bestCost: " << bestCost << BR << endl;
  const BoolVector noShift(pEnd,false);
  bool isCostSearch = true;  /* search only on players that cause non-zero cost function */
  for (int d = 1; pBegin < pEnd && d <= depth; ++d) {
    //cout << "depth=" << depth << BR << endl;
    IndexVector nextPair = bestPair;
    IndexSet nextCostPlayers = bestCostPlayers;
    Cost nextCost = bestCost;
    IndexVector i(2*d, pBegin);
    // find next best pairing with at most d player swaps
    int testNum = 0;
#define GREEDY_SEARCH	1
#if GREEDY_SEARCH
    bool isFoundBetter = false;
#endif
    while (!bestCost.IsZero()) {
      // d is number of swaps
      // j is 0 or 1 (first or second of the swap) for d=1
      // i[j]/2 is like the board number if boards started at #0 (two players on each board)
      // pair[i[j]] is the rank of the player
      // pl[pair[i[j]]] is the player
      nextI:
      for (size_t j = 0; j < i.size() && (++i[j] >= pEnd || pl[bestPair[i[j]]].play_id == BYE_ID); ++j)
        i[j] = pBegin;
      if (i == IndexVector(2*d,pBegin))
        break;  // wrap around, so done
      for (size_t j = 0; j < i.size(); j += 2) {
        if ((j > 0 && (d <= 1 ? i[j] <= i[j-2] : i[j] < i[j-2])) || (d <= 1 ? i[j+1] <= i[j] : i[j+1] < i[j]))
          goto nextI;  // don't do things twice
        if (isCostSearch && bestCostPlayers.find(bestPair[i[j]]) == bestCostPlayers.end() && bestCostPlayers.find(bestPair[i[j+1]]) == bestCostPlayers.end())
          goto nextI;
      }

      size_t maxChange = 0;
      for (size_t j = 0; j < i.size(); j += 2) {
        ASSERT(d <= 1 ? i[j+1] > i[j] : i[j+1] >= i[j]);
        if (maxChange < i[j+1] - i[j])
          maxChange = i[j+1] - i[j];
      }
      for (size_t s = 0; s < (maxChange <= 2 ? 1 : 8); ++s) {
#if PERF_DEBUG
        ASSERT(sTry.size() == sDo.size() && sDo.size() > s);
#endif /* PERF_DEBUG */
        // try simple swap (s=0) or more-complex rotate (s>0)
        IndexVector testPair = bestPair;
        //cout << "s=" << s << " i: " << i << BR << endl;
        for (size_t j = 0; j < i.size(); j += 2) {
          if (i[j] >= i[j+1]) {
            ASSERT(d >= 2 && i[j] == i[j+1]);
            continue;
          }
          const bool hasBye2 = (hasBye && (i[j] >= pEnd-2 || i[j+1] >= pEnd-2));
          const size_t pEnd2 = (hasBye && !hasBye2 ? pEnd-2 : pEnd);
          if (s == 0) {
#ifdef OLD_CODE
            cout << testPair[i[j]] << ' ' << testPair[i[j+1]] << BR << endl;
            if ((pl[testPair[i[j]]].rating == 116 || pl[testPair[i[j+1]]].rating == 116)
		&& (pl[testPair[i[j]]].uscf_id == 15358962 || pl[testPair[i[j+1]]].uscf_id == 15358962)) {
              cout << pl[testPair[i[j]]] << BR << pl[testPair[i[j+1]]] << BR << endl;
              cout << testPair << BR << endl;
            }
#endif /* OLD_CODE */
            swap(testPair[i[j]], testPair[i[j+1]]);
#ifdef OLD_CODE
            if ((pl[testPair[i[j]]].rating == 116 || pl[testPair[i[j+1]]].rating == 116)
		&& (pl[testPair[i[j]]].uscf_id == 15358962 || pl[testPair[i[j+1]]].uscf_id == 15358962)) {
              cout << testPair << BR << endl;
            }
#endif /* OLD_CODE */
          } else if (s == 1) {
            ASSERT(pl[testPair[i[j+1]]].play_id != BYE_ID);
            //cout << "calling RotatePairDown()" << " i[j+1]=" << i[j+1] << BR << endl;
            //cout << "testPair: " << testPair << BR << endl;
            RotatePairDown(testPair, i[j], i[j+1], pBegin, pEnd2, hasBye2, false, noShift);
          } else if (s == 2) {
            RotatePairUp(testPair, i[j], i[j+1], pBegin, pEnd2, hasBye2, false, noShift);
          } else if (s == 3 || s == 4 || s == 5) {
            // these cases only rotate within a score group (might include a few stragglers for multiple drop down and/or multiple pull up)
            //cout << " pBegin=" << pBegin << " i[j]=" << i[j] << " i[j+1]=" << i[j+1] << " pEnd2=" << pEnd2 << BR << endl;
            //cout << pl[testPair[i[j]]] << BR << endl;
            //cout << pl[testPair[i[j+1]]] << BR << endl;
            const real score = pl[testPair[i[j]]].score;
            if (pl[testPair[i[j+1]]].score != score) goto nextS;
            //cout << "loop" << endl;
            size_t sBegin, sEnd;
            for (sBegin = i[j]/2*2; sBegin > pBegin && pl[testPair[sBegin-2]].score == score && pl[testPair[sBegin-1]].score == score; sBegin -= 2)
              ;//cout << sBegin << endl;
            const bool oddPullUp = (i[j] == sBegin+1 && pl[testPair[sBegin]].score > score);
            for (sEnd = i[j+1]/2*2+2; sEnd < pEnd2 && pl[testPair[sEnd]].score == score && pl[testPair[sEnd+1]].score == score; sEnd += 2)
              ;//cout << sEnd << endl;
            const bool oddDropDown = (i[j+1] == sEnd-2 && (pl[testPair[sEnd-1]].score < score || pl[testPair[sEnd-1]].play_id == BYE_ID));
            //cout << " pBegin=" << pBegin << " sBegin=" << sBegin << " i[j]=" << i[j] << " i[j+1]=" << i[j+1] << " sEnd=" << sEnd << " pEnd2=" << pEnd2 << " oddDropDown=" << oddDropDown << " oddPullUp=" << oddPullUp << endl;
            ASSERT(pBegin <= sBegin && sBegin <= i[j] && i[j] < i[j+1] && i[j+1] <= sEnd && sEnd <= pEnd2);
            ASSERT(!hasBye2 || sEnd == pEnd2);
            if (s == 3) {
              RotatePairDown(testPair, i[j], i[j+1], sBegin, sEnd, oddDropDown, oddPullUp, noShift);
            } else if (s == 4) {
              RotatePairUp(testPair, i[j], i[j+1], sBegin, sEnd, oddDropDown, oddPullUp, noShift);
            } else {
              bool isChanged = RotateColor(pl, testPair, i[j], i[j+1], sBegin, sEnd, oddDropDown, oddPullUp);
              if (!isChanged)
                goto nextS;
            }
          } else if (s == 6 || s == 7) {
            ASSERT(pl[testPair[i[j+1]]].play_id != BYE_ID);
            BoolVector shift(pEnd2,false);
            const char startColor = AllocateColor(pl[testPair[pBegin]], pl[testPair[pBegin%2==0?pBegin+1:pBegin-1]], (pBegin/2%2 == 0));
            for (size_t c = pBegin/2*2 + 2; c < pEnd2; c += 2)
              shift[c] = (startColor != AllocateColor(pl[testPair[c]], pl[testPair[c+1]], (c/2%2==0)));
            //cout << "calling RotatePairDown()" << " i[j+1]=" << i[j+1] << BR << endl;
            //cout << "testPair: " << testPair << BR << endl;
            if (s == 6)
              RotatePairDown(testPair, i[j], i[j+1], pBegin, pEnd2, hasBye2, false, shift);
            else
              RotatePairUp(testPair, i[j], i[j+1], pBegin, pEnd2, hasBye2, false, shift);
          } else {
            ASSERT(0);
          }
        }
        for (size_t y = 0; y < testPair.size(); y += 2) {
          // don't put ranks out of order
          if (testPair[y] >= testPair[y+1]) {
#ifdef OLD_CODE
            if (s == 0)
              goto nextS;
            else
#endif
            swap(testPair[y], testPair[y+1]);
          }
        }
        {
          ++testNum;
          //cout << "s=" << s << " i: " << i << BR << endl;
          //cout << "testPair: " << testPair << BR << endl;
          SortBoards(pl, testPair);
          //cout << "testNum=" << testNum << " testPair: " << testPair << BR << endl;
          IndexSet testCostPlayers;
          const Cost testCost = CostFunction(pl, testPair, remainingRounds, pBegin, pEnd, false, usePairableCost, testCostPlayers);
          //cout << "testNum=" << testNum << " testCost: " << testCost << " testCostPlayers: " << testCostPlayers << BR << endl;

#if GREEDY_SEARCH
          if (testCost < bestCost) {
#if PERF_DEBUG
            ++sDo[s];
#endif /* PERF_DEBUG */
            nextPair = bestPair = testPair;
            nextCost = bestCost = testCost;
            nextCostPlayers = bestCostPlayers = testCostPlayers;
            isFoundBetter = true;
          }
#else /* GREEDY_SEARCH */
          if (testCost < nextCost) {
#ifdef PERF_DEBUG
            ++sDo[s];
#endif /* PERF_DEBUG */
            nextPair = testPair;
            nextCost = testCost;
            nextCostPlayers = bestCostPlayers = testCostPlayers;
            //cout << "nextPair: " << nextPair << BR << endl;
            //cout << "nextCost: " << nextCost << BR << endl;
            //cout << "nextCostPlayers: " << nextCostPlayers << BR << endl;
            //break;  // TBD: does this make the optimizer faster?
          }
#endif /* GREEDY_SEARCH */
        }
#if PERF_DEBUG
        ++sTry[s];
#endif /* PERF_DEBUG */
        nextS:;
      }
    }
#if GREEDY_SEARCH
    if (isFoundBetter) {
      --d;  // look for something even better
      //isCostSearch = true;  // redo with this flag
#ifdef OLD_CODE
    } else if (isCostSearch) {
      isCostSearch = false;
      --d;  // look at all possibilities
#endif /* OLD_CODE */
    }
#else /* GREEDY_SEARCH */
    // save next best pairing and loop again with same d, but only if better
    if (nextCost < bestCost) {
      bestPair = nextPair;
      bestCost = nextCost;
      bestCostPlayers = nextCostPlayers;
      --d;
      //isCostSearch = true;  // redo with this flag
#ifdef OLD_CODE
    } else if (isCostSearch) {
      isCostSearch = false;
      --d;  // look at all possibilities
#endif /* OLD_CODE */
    }
#endif /* GREEDY_SEARCH */
    //cout << "bestPair: " << bestPair << BR << endl;
    //cout << "bestCost: " << bestCost << BR << endl;
    //cout << "bestCostPlayers: " << bestCostPlayers << BR << endl;
  }
  pair = bestPair;
  
#if USE_PAIRABLE_COST
  if (USE_PAIRABLE_COST && !usePairableCost) {
    const Cost c = CostFunction(pl, pair, remainingRounds, pBegin, pEnd, false, true);
    if (c != bestCost) {
      // redo using PairableCost
#if PERF_DEBUG
      cout << "sec_id=" << pl[0].sec_id << ": MinimizePairingCost() redo: costCount=" << costCount;
      for (size_t x = 0; x < sTry.size() && x < sDo.size(); ++x) cout << " sTry[" << x << "]=" << sTry[x] << " sDo[" << x << "]=" << sDo[x];
      cout << BR << endl;
#endif
      //cout << "redo using PairableCost()"BR << endl;
      //cout << bestCost << BR << endl;
      //cout << c << BR << endl;
      //exit(-1);
      return MinimizePairingCost(pl, pair, remainingRounds, depth, pBegin, pEnd, true);
    }
  }
#endif /* USE_PAIRABLE_COST */
  const Cost c = CostFunction(pl, pair, remainingRounds, pBegin, pEnd, true, true);  // should be same as bestCost, but need to setup warn_codes
#if DEBUG
  cout << "done MinimizePairingCost()"BR << endl;
  AssertNoDuplicates(pl, pair);
  cout << c << BR << endl;
#endif
#if PERF_DEBUG
  cout << "sec_id=" << pl[0].sec_id << ": MinimizePairingCost(): costCount=" << costCount;
  for (size_t x = 0; x < sTry.size() && x < sDo.size(); ++x) cout << " sTry[" << x << "]=" << sTry[x] << " sDo[" << x << "]=" << sDo[x];
  cout << BR << endl;
  cout << "End time: " << flush; system("date"); cout << BR << endl;
#endif
  return c;
}

void SetRanks (PlayerVector &pl)
{
  //cout << "SetRanks()"BR << endl;
  typedef map<integer,integer> MapII;
  MapII rankMap;
  for (size_t x = 0; x < pl.size(); ++x) {
    ASSERT(x == pl.size()-1 ? pl[x].play_id == BYE_ID : pl[x].play_id != BYE_ID);
    pl[x].rank = x;
    rankMap.insert(pair<integer,integer>(pl[x].play_id, x));
    //cout << pl[x] << BR << endl;
    pl[x].due_color = DueColor(pl[x].color_history, pl[x].multiround);  // assigns 'x' for BYE_ID
    //cout << "pl[" << x << "].opponents=" << pl[x].opponents << BR << endl;
    //cout << "pl[" << x << "].opponent_ranks=" << pl[x].opponent_ranks << BR << endl;
    //cout << "pl[" << x << "].teammates=" << pl[x].teammates << BR << endl;
    //cout << "pl[" << x << "].teammate_ranks=" << pl[x].teammate_ranks << BR << endl;
  }
  for (size_t x = 0; x < pl.size(); ++x) {
    //cout << "x=" << x << BR << endl;
    pl[x].opponent_ranks.clear();
    for (size_t y = 0; y < pl[x].opponents.size(); ++y) {
      const MapII::const_iterator op = rankMap.find(I(pl[x].opponents[y]));  // I() extracts only play_id
      if (op != rankMap.end()) {
        ASSERT(op != rankMap.end());
        pl[x].opponent_ranks.push_back(op->second);
      }
    }
    pl[x].teammate_ranks.clear();
    for (size_t y = 0; y < pl[x].teammates.size(); ++y) {
      const MapII::const_iterator tm = rankMap.find(pl[x].teammates[y]);
      if (tm != rankMap.end())
        pl[x].teammate_ranks.push_back(tm->second);
    }
    //cout << "x=" << x << " pl[" << x << "].opponent_ranks=" << pl[x].opponent_ranks << BR << endl;
    //cout << "x=" << x << " pl[" << x << "].teammate_ranks=" << pl[x].teammate_ranks << BR << endl;
  }
}

void CanonicalPlayerVector (PlayerVector &pl)
{
#if DEBUG
  cout << "CanonicalPlayerVector(" << pl.size() << ")"BR << endl;
#endif
  if (pl.size() <= 0 || pl.back().play_id != BYE_ID) {
    pl.resize(pl.size()+1);
    pl.back().play_id = BYE_ID;
    pl.back().board_num = -1;
    pl.back().bye_request = false;
    pl.back().paired = false;
    pl.back().rnd = pl[0].rnd;
    pl.back().multiround = pl[0].multiround;
  }
  sort(pl.begin(), pl.end());
  SetRanks(pl);
  ASSERT(pl.back().play_id == BYE_ID);
  for (size_t x = 0; x < pl.size()-1; ++x)
    ASSERT(pl[x].play_id != BYE_ID);
#if DEBUG
  cout << "end CanonicalPlayerVector(" << pl.size() << ")"BR << endl;
#endif
}

bool LessRobinSort (const Player &x, const Player &y)
{
  const bool byeX = (x.play_id == BYE_ID);
  const bool byeY = (y.play_id == BYE_ID);
  return byeX < byeY || (byeX == byeY && x.rand < y.rand);
}

// depth==1 takes a few seconds; depth==2 takes a minute on a small section; depth > 2 takes a long time
Cost FindPairings (PlayerVector &pl, smallint totalRounds, integer firstBoardNum, int depth, bool useFirstPairings, bool skipOptimize, const string &secName)
{
#if DEBUG
  cout << "FindPairings(" << pl.size() << ")"BR << endl;
#endif
  if (pl.size() <= 1) {
    cout << "WARNING: nobody active to pair in " << secName << BR << endl;
  } else if (pl[0].multiround != 1) {
    const smallint mr = pl[0].multiround;
    for (size_t x = 0; x < pl.size(); ++x) {
      const Player &px = pl[x];
      ASSERT(px.multiround == mr);
      for (size_t y = 0; y < px.opponents.size(); y += mr) {
        const text opponent = px.opponents[y];
        for (size_t z = y; z < y+mr && z < px.opponents.size(); ++z) {
          if (px.opponents[z] != opponent) {
            cout << "<font color=red>ERROR: not same opponents across multiround</font>" << BR << px << BR << endl;
            break;
          }
        }
      }
    }
  }

  // request bye for one odd house player
  int housePlayer = -1;
  size_t players = 0;
  for (size_t x = 0; x < pl.size(); ++x) {
    if (!pl[x].bye_request && !pl[x].paired && pl[x].play_id != BYE_ID) {
      ++players;
      if (pl[x].bye_house)
        housePlayer = x;
    }
  }
  if (players % 2 == 0)
    housePlayer = -1;
  if (housePlayer >= 0) {
    cout << "INFO: requesting bye for house player, " << pl[housePlayer].player_name << BR << endl;
    pl[housePlayer].bye_request = true;  // odd house player requests bye
    --players;
  }

  // put PlayerVector in canonical form (sorted with bye at end)
  CanonicalPlayerVector(pl);
  //cout << "pl=";
  //for (size_t x = 0; x < pl.size(); ++x)
    //cout << pl[x].play_id << '_' << pl[x].reentry
	//<< '(' << pl[x].score << ',' << pl[x].rating << ')';
  //cout << BR << endl;

  // short-cut for round robin pairings
  if (pl.size() > 0 && (pl[0].trn_type == 'R' || pl[0].trn_type == 'D')) {
    sort(pl.begin(), pl.end(), LessRobinSort);
    totalRounds /= pl[0].multiround;
    //cout << "pl.size()=" << pl.size() << " totalRounds=" << totalRounds << BR << endl;
    ASSERT(int(pl.size()) - 1 == totalRounds);
    size_t withdrawnPlayer = 0;
    for (size_t x = 0; x < pl.size(); ++x) {
      const Player &px = pl[x];
      ASSERT(px.play_id != BYE_ID || x == pl.size()-1);
      if (px.bye_rounds.size() > 0 && px.bye_rounds[0] <= (totalRounds+1)/2) {
        ASSERT(withdrawnPlayer == 0);
        withdrawnPlayer = x+1;
      }
    }
    for (size_t x = 0; x < pl.size(); ++x) {
      Player &px = pl[x];
      //cout << "before Crenshaw"BR << endl;
      CrenshawBergerLookup(pl.size(), (px.rnd-1)/px.multiround+1, x+1, withdrawnPlayer, px.board_num, px.board_color);
      //cout << "after Crenshaw"BR << endl;
      px.board_num += firstBoardNum - 1;
      //cout << "x=" << x << " board=" << px.board_num << " color=" << px.board_color << " firstBoardNum=" << firstBoardNum << BR << endl;
    }
    if (pl.back().play_id == BYE_ID) {
      for (size_t x = 0; x < pl.size()-1; ++x) {
        if (pl[x].board_num == pl.back().board_num) {
          pl[x].board_color = 'W';
          pl.back().board_color = 'B';
          break;
        }
      }
      ASSERT(pl.back().board_color == 'B');
    }
    //cout << "Done with Round Robin pairings"BR << endl;
    return Cost();
  }

  // compute ranks, rank lists, due colors, players who want pairings, and lowest board
  integer lowBoard = INT_MAX;
  for (size_t x = 0; x < pl.size(); ++x) {
    ASSERT(x == pl.size()-1 ? pl[x].play_id == BYE_ID : pl[x].play_id != BYE_ID);
    if (pl[x].play_id != BYE_ID && lowBoard > pl[x].board_num)
      lowBoard = pl[x].board_num;
  }
  if (firstBoardNum == 0)
    firstBoardNum = lowBoard;

  // find starting point (for all players)
  IndexVector pair;
  HintPairings(pl, pair, true);	// calculate base situation using given board assignments as hint
  //cout << "pair=" << flush;
  //for (size_t x = 0; x < pair.size(); ++x)
    //cout << ' ' << pair[x] << (x%2==0?":W:":":B:") << pl[pair[x]].play_id << '_' << pl[pair[x]].reentry;
  //cout << BR << endl;
  if (players > 0) {
    //cout << "first check players=" << players << BR << endl;
    Player &p = pl[pair[players-1]];
    //cout << p << BR << endl;
    ASSERT(!p.bye_request && !p.paired);
  }
  //cout << "done first check"BR << endl;
  if (players < pair.size() && pl[pair[players]].play_id != BYE_ID) {
    //cout << "second check"BR << endl;
    Player &p = pl[pair[players]];
    //cout << p << BR << endl;
    ASSERT(p.bye_request || p.paired);
  }
  //cout << "done second check"BR << endl;

  if (useFirstPairings)
    FirstPairings(pl, pair, players, totalRounds);	// calculate base situation without conflicts for active non-bye players (ignoring hint)
  //cout << "pair=";
  //for (size_t x = 0; x < pair.size(); ++x)
    //cout << (x%2==0?" W:":" B:") << pl[pair[x]].play_id << '_' << pl[pair[x]].reentry;
  //cout << BR << endl;
  //cout << "players=" << players << BR << endl;
  if (players > 0) {
    //cout << "third check"BR << endl;
    Player &p = pl[pair[players-1]];
    //cout << p << BR << endl;
    ASSERT(!p.bye_request && !p.paired);
  }
  //cout << "done third check"BR << endl;
  if (players < pair.size() && pl[pair[players]].play_id != BYE_ID) {
    //cout << "fourth check"BR << endl;
    Player &p = pl[pair[players]];
    //cout << p << BR << endl;
    ASSERT(p.bye_request || p.paired);
  }
  //cout << "done fourth check"BR << endl;

#ifdef OLD_CODE
  // undo bye request for odd house player
  if (housePlayer >= 0) {
    pl[pair[housePlayer]].bye_request = false;
    ++players;
  }
#endif /* OLD_CODE */

  const Cost cost = (skipOptimize ?
	CostFunction(pl, pair, totalRounds - pl[0].rnd, 0, (players+1)/2*2, true, true) :
	MinimizePairingCost(pl, pair, totalRounds - pl[0].rnd, depth, 0, players, false));

  // set boards and colors (active gets lower boards)
  //cout << "set boards and colors"BR << endl;
  ASSERT(pair.size() % 2 == 0);
  // first sort by rank (putting byes last)
  for (size_t x = 2; x < pair.size(); x += 2) {
    for (size_t y = x; y > 0; y -= 2) {
      const size_t z1 = (pl[pair[y-2]] < pl[pair[y-1]] ? y-2 : y-1);
      const size_t z2 = (pl[pair[y]] < pl[pair[y+1]] ? y : y+1);
      const bool b1 = (pl[pair[y-2]].play_id == BYE_ID || pl[pair[y-1]].play_id == BYE_ID);
      const bool b2 = (pl[pair[y]].play_id == BYE_ID || pl[pair[y+1]].play_id == BYE_ID);
      if (b1 < b2 || (b1 == b2 && pl[pair[z1]] < pl[pair[z2]]))
        break;
      swap(pair[y], pair[y-2]);
      swap(pair[y+1], pair[y-1]);
    }
  }
  //cout << "pair=" << pair << BR << endl;
  // set boards
  for (size_t x = 0; x < pair.size(); x += 2) {
    //cout << x << endl;
    //ASSERT(x == 0 || pl[pair[x-2]] < pl[pair[x]] || (pl[pair[x-1]].play_id != BYE_ID && pl[pair[x+1]].play_id == BYE_ID));  // boards sorted
    ASSERT(pl[pair[x]].play_id != BYE_ID);
    pl[pair[x]].board_num =
	pl[pair[x+1]].board_num = firstBoardNum + x / 2;
    //cout << "x: " << pl[pair[x]] << BR << endl;
    //cout << "x+1: " << pl[pair[x+1]] << BR << endl;
    pl[pair[x]].board_color = FlipColor(
	pl[pair[x+1]].board_color = AllocateColor(pl[pair[x+1]], pl[pair[x]], x/2%2==0)
	);
    //cout << pl[pair[x]] << BR << pl[pair[x+1]] << BR << endl;
    ASSERT(pl[pair[x]].board_num == pl[pair[x+1]].board_num);
    ASSERT((pl[pair[x]].board_color == 'W' && pl[pair[x+1]].board_color == 'B') || (pl[pair[x]].board_color == 'B' && pl[pair[x+1]].board_color == 'W'));
  }
  //cout << BR << endl;
  // set colors
  for (size_t x = 0; x < pl.size(); ++x) {
    ASSERT(pl[x].board_color == 'W' || pl[x].board_color == 'B' || pl[x].play_id == BYE_ID);
    //cout << pl[x] << BR << endl;
    for (size_t y = 0; y < pair.size(); y += 2) {
      if (pl[pair[y+1]].play_id == BYE_ID)
        continue;
      ASSERT(pl[pair[y]].board_num == pl[pair[y+1]].board_num);
      if ((pl[pair[y]].play_id == pl[x].play_id && pl[pair[y]].reentry == pl[x].reentry)
		|| (pl[pair[y+1]].play_id == pl[x].play_id && pl[pair[y+1]].reentry == pl[x].reentry))
        continue;
      ASSERT(pl[pair[y]].board_num != pl[x].board_num);
    }
  }
  ASSERT(pl.back().play_id == BYE_ID);
  pl.back().board_num = -1;
  //if (pl.back().play_id == BYE_ID && pl.back().board_num == -1)
    //pl.pop_back();
  //cout << "done with boards and colors"BR << endl;

  return cost;
}

////////////////////////  TIEBREAK FUNCTIONS  ////////////////////////

//#include <numeric>

struct PlayerResult
{
  /* INPUT FIELDS - provided by function caller; treated as constants */

  text player;  // key by which this value is indexed (duplicate of the map key)
  smallint rating;

// need tiebreak order to determine which are the top players to add up the scores for team tiebreaks
// so we won't calculate team tiebreaks here
// http://hpchess.org/tie-breaks/

  // these three vectors have the same size, the number of rounds for this player
  // all players don't necessarily play the same number of rounds, like when calculating scores across sections
  // withdrawn players should have 'U' for games missed so that the number of games matches their section
  textVector opponent;	// opponent key for each round in order
  charVector color;	// color for each round in order
  charVector result;	// result for each round in order

  /* CALCULATION FIELDS - should be ignored by the caller, both input and output */

  real rawScore, adjScore, cumScore, byeScore, head2head;
#if MATCH_SWISS_SYS
  real cumScore2;
#endif
  smallint byeCnt, blackCnt, kashdan, winCnt;
  size_t firstLossRound;
  double performanceRating;
  double coinFlip;

  /* OUTPUT FIELDS - calculated by the function, to be used by the caller */

  charVector tiebreak_code;		// type of tie break (A through Z)
  doubleVector tiebreak_value;		// value of tie break, same order as above
};

ostream &operator<< (ostream &out, const PlayerResult &pr)
{
  return out
	<< "player=" << pr.player
	<< " rating=" << pr.rating
	<< " opponent=" << pr.opponent
	<< " color=" << pr.color
	<< " result=" << pr.result
	<< " rawScore=" << pr.rawScore
	<< " adjScore=" << pr.adjScore
	<< " byeScore=" << pr.byeScore
	<< " head2head=" << pr.head2head
	<< " byeCnt=" << pr.byeCnt
	<< " blackCnt=" << pr.blackCnt
	<< " kashdan=" << pr.kashdan
	<< " winCnt=" << pr.winCnt
	<< " firstLossRound=" << pr.firstLossRound
	<< " performanceRating=" << pr.performanceRating
	<< " coinFlip=" << fixed << pr.coinFlip
	<< " tiebreak_code=" << pr.tiebreak_code
	<< " tiebreak_value=" << pr.tiebreak_value
	;
}

typedef map<text, PlayerResult> PlayerResultMap;  // indexed by player, which is typically two numbers separated by '_' (player_id and reentry)
typedef PlayerResultMap::iterator PRMiter;
typedef PlayerResultMap::const_iterator cPRMiter;
typedef pair<text, PlayerResult> PlayerResultElem;

void TiebreakPlayer (PlayerResult &p, const text &byeKey)
{
  const size_t rounds = p.color.size();
  ASSERT(p.opponent.size() == rounds && p.color.size() == rounds && p.result.size() == rounds);
  p.rawScore = 0;
  p.adjScore = 0;
  p.cumScore = 0;
  p.byeScore = 0;
  p.kashdan = 0;
  p.byeCnt = 0;
  p.blackCnt = 0;
  p.winCnt = 0;
  p.firstLossRound = 0;
  p.coinFlip = -1;
  if (p.player == byeKey) {
    ASSERT(p.rating == 0);
    return;
  }
  for (size_t x = 0; x < rounds; ++x) {
    p.firstLossRound += (p.firstLossRound == x);
    switch (p.result[x]) {
    case '$':				p.rawScore += 2.0; p.adjScore += 2.0; p.byeScore += 0.0; p.kashdan += 4+4; ++p.blackCnt; p.winCnt += 2; break;
    case '#':				p.rawScore += 1.5; p.adjScore += 1.5; p.byeScore += 0.0; p.kashdan += 4+2; ++p.blackCnt; ++p.winCnt; break;
    case '%':				p.rawScore += 1.0; p.adjScore += 1.0; p.byeScore += 0.0; p.kashdan += 2+2; ++p.blackCnt; break;
    case 'W': case 'N':			p.rawScore += 1.0; p.adjScore += 1.0; p.byeScore += 0.0; p.kashdan += 4; p.blackCnt += (p.color[x]=='B'); ++p.winCnt; break;
    case 'B': case 'X':			p.rawScore += 1.0; p.adjScore += 0.5; p.byeScore += 1.0; p.kashdan += 0; ++p.byeCnt; break;
    case 'D': case 'R':			p.rawScore += 0.5; p.adjScore += 0.5; p.byeScore += 0.0; p.kashdan += 2; p.blackCnt += (p.color[x]=='B'); break;
    case 'H': case 'Z':			p.rawScore += 0.5; p.adjScore += 0.5; p.byeScore += 0.5; p.kashdan += 0; ++p.byeCnt; break;
    case 'L': case 'S':			p.rawScore += 0.0; p.adjScore += 0.0; p.byeScore += 0.0; p.kashdan += 1; p.blackCnt += (p.color[x]=='B'); p.firstLossRound -= (p.firstLossRound == x+1); break;
    case 'U': case 'F': case '*':	p.rawScore += 0.0; p.adjScore += 0.5; p.byeScore += 0.0; p.kashdan += 0; ++p.byeCnt; break;
    default: ASSERT(0);
    }
    p.cumScore += p.rawScore;
  }
#if MATCH_SWISS_SYS
  p.cumScore2 = p.cumScore;
#endif
  p.cumScore -= p.byeScore;
  ++p.firstLossRound;  // between 1 and N+1 instead of between 0 and N
}

void TiebreakCoinFlip (const PlayerResultMap &prm, PlayerResult &p, const text &byeKey)
{
  if (p.player == byeKey)
    return;
  static bool init = false;
  if (!init) {
    srand(time(0));
    init = true;
  }
  nextFlip:
  p.coinFlip = rand();
  for (cPRMiter i = prm.begin(); i != prm.end() && i->second.player != p.player; ++i)
    if (i->second.coinFlip == p.coinFlip)
      goto nextFlip;  // make sure there are no ties
}

void TiebreakPerformance (const PlayerResultMap &prm, PlayerResult &p, const text &byeKey)
{
  const size_t rounds = p.color.size();
  size_t playerCnt = 0;
  double ratingSum = 0;
  p.head2head = 0;
  ASSERT(p.rawScore * 2 == int(p.rawScore * 2));
  for (size_t x = 0; p.player != byeKey && x < rounds; ++x) {
    const PlayerResult &opponent = prm.find(p.opponent[x])->second;
    ASSERT(opponent.player != p.player);
    ASSERT(opponent.rawScore * 2 == int(opponent.rawScore * 2));
    if (opponent.rawScore == p.rawScore) {
      // Result between tied players, rule 34E5
      switch (p.result[x]) {
      case '$':				p.head2head += 2.0-0.0; continue;
      case '#':				p.head2head += 1.5-0.5; continue;
      case '%':				p.head2head += 0.5-0.5; continue;
      case 'W': case 'N':		p.head2head += 1.0-0.0; continue;
      case 'B': case 'X':		continue;
      case 'D': case 'R':		p.head2head += 0.5-0.5; continue;
      case 'H': case 'Z':		continue;
      case 'L': case 'S':		p.head2head += 0.0-1.0; continue;
      case 'U': case 'F': case '*':	continue;
      default: ASSERT(0);
      }
    } else {
      // Opposition's performance, rule 34E10
      switch (p.result[x]) {
      case '$':				ratingSum += 400; break;
      case '#':				ratingSum += 200; break;
      case '%':				ratingSum += 0; break;
      case 'W': case 'N':		ratingSum += 400; break;
      case 'B': case 'X':		continue;
      case 'D': case 'R':		ratingSum += 0; break;
      case 'H': case 'Z':		continue;
      case 'L': case 'S':		ratingSum += -400; break;
      case 'U': case 'F': case '*':	continue;
      default: ASSERT(0);
      }
      ratingSum += opponent.rating;
      ++playerCnt;
    }
  }
  p.performanceRating = (playerCnt <= 0 ? p.rating : ratingSum / playerCnt);
}

bool TiebreakPlayed (char result)
{
  switch (result) {
  case 'B': case 'X':
  case 'H': case 'Z':
  case 'U': case 'F': case '*':
    return false;
  default:
    return true;
  }
}

void TiebreakOpponent (const PlayerResultMap &prm, PlayerResult &p, const text &byeKey)
{
  const size_t rounds = p.color.size();
  realVector adj;
  adj.reserve(rounds);
  double adjSum = 0, cumSum = 0;
  double ratSum = 0, perfSum = 0;
  real partialScore = 0;
  size_t playCnt = 0;
  //cout << p.player << " adj:";
  for (size_t x = 0; p.player != byeKey && x < rounds; ++x) {
    const PlayerResult &opponent = prm.find(p.opponent[x])->second;
    const bool isPlayed = TiebreakPlayed(p.result[x]);
    adjSum += isPlayed * opponent.adjScore;
#if MATCH_SWISS_SYS
    cumSum += opponent.cumScore2;
#else
    cumSum += opponent.cumScore;
#endif
    adj.push_back(isPlayed * opponent.adjScore);
    //cout << ' ' << adj.back();
    if (isPlayed) {
      ++playCnt;
      ratSum += opponent.rating;
      perfSum += opponent.performanceRating;
    }
    switch (p.result[x]) {
    case '$':				partialScore += opponent.rawScore + opponent.rawScore;
    case '#':				partialScore += opponent.rawScore + opponent.rawScore / 2;
    case '%':				partialScore += opponent.rawScore / 2 + opponent.rawScore / 2;
    case 'W': case 'N':			partialScore += opponent.rawScore;
    case 'B': case 'X':			break;
    case 'D': case 'R':			partialScore += opponent.rawScore / 2;
    case 'H': case 'Z':			break;
    case 'L': case 'S':			break;
    case 'U': case 'F': case '*':	break;
    default: ASSERT(0);
    }
  }
  //cout << BR << endl;
  sort(adj.begin(), adj.end());
  const double ratAvg = (playCnt <= 0 ? p.rating : ratSum / playCnt);
  const double perfAvg = (playCnt <= 0 ? p.performanceRating : perfSum / playCnt);

  // http://en.wikipedia.org/wiki/Tie-breaking_in_Swiss-system_tournaments
  p.tiebreak_code.push_back('M');	// Modified median Harkness, rule 34E1
  p.tiebreak_value.push_back(
	rounds < 2 ?		0 :
	rounds < 9 ?		adjSum - (round(p.rawScore * 2) >= rounds) * adj.front() - (round(p.rawScore * 2) <= rounds) * adj.back() :
				adjSum - (round(p.rawScore * 2) >= rounds) * (adj[0] + adj[1]) - (round(p.rawScore * 2) <= rounds) * (adj[rounds-2] + adj[rounds-1])
	);
  p.tiebreak_code.push_back('S');	// Solkoff, rule 34E2
  p.tiebreak_value.push_back(adjSum);
  p.tiebreak_code.push_back('C');	// Cumulative score, rule 34E3
  p.tiebreak_value.push_back(p.cumScore);
  p.tiebreak_code.push_back('B');	// basic median system not modified, rule 34E4
  p.tiebreak_value.push_back(
	rounds <= 2 ?		0 :
	rounds < 9 ?		adjSum - adj.front() - adj.back() :
				adjSum - adj[0] - adj[1] - adj[rounds-2] - adj[rounds-1]
	);
  p.tiebreak_code.push_back('H');	// Head-to-head result between tied players, rule 34E5
  p.tiebreak_value.push_back(p.head2head);
  p.tiebreak_code.push_back('T');	// Total blacks, rule 34E6
  p.tiebreak_value.push_back(p.blackCnt);
  p.tiebreak_code.push_back('K');	// Kashdan aggressive, rule 34E7
  p.tiebreak_value.push_back(p.kashdan);
  p.tiebreak_code.push_back('R');	// Round robin Sonneborn-Berger, rule 34E8
  p.tiebreak_value.push_back(partialScore);
  p.tiebreak_code.push_back('O');	// Opposition cumulative score, rule 34E9
  p.tiebreak_value.push_back(cumSum);
  p.tiebreak_code.push_back('P');	// Performance of opposition, rule 34E10
  p.tiebreak_value.push_back(perfAvg);
  p.tiebreak_code.push_back('A');	// Average rating of opposition, rule 34E11
  p.tiebreak_value.push_back(ratAvg);
  p.tiebreak_code.push_back('W');	// Win count
  p.tiebreak_value.push_back(p.winCnt);
  p.tiebreak_code.push_back('L');	// First loss round
  p.tiebreak_value.push_back(p.firstLossRound);
					// no calculation for Speed play-off game, rule 34E12
  p.tiebreak_code.push_back('Z');	// Coin flip, rule 34E13
  //cout << "CoinFlip=" << fixed << p.coinFlip << BR << endl;
  p.tiebreak_value.push_back(p.coinFlip);
}

// need tiebreak order to determine which are the top players to add up the scores for team tiebreaks
// http://hpchess.org/tie-breaks/

void TiebreakCalculation (PlayerResultMap &prm, const text &byeKey)
{
#if DEBUG
  cout << "TiebreakCalculation(prm,byeKey=" << byeKey << ")"BR << endl;
#endif
  // individual tiebreak calculations
  PlayerResult *bye = 0;
  for (PRMiter i = prm.begin(); i != prm.end(); ++i) {
    PlayerResult &p = i->second;
    //cout << p << BR << endl;
    ASSERT(i->first == p.player);
    TiebreakPlayer(p, byeKey);
    TiebreakCoinFlip(prm, p, byeKey);
    if (p.player == byeKey) {
      ASSERT(bye == 0);
      bye = &i->second;
    }
  }
  ASSERT(bye != 0);
  ASSERT(bye->player == byeKey);
  for (PRMiter i = prm.begin(); i != prm.end(); ++i) {
    PlayerResult &p = i->second;
    TiebreakPerformance(prm, p, byeKey);
  }
  for (PRMiter i = prm.begin(); i != prm.end(); ++i) {
    PlayerResult &p = i->second;
    TiebreakOpponent(prm, p, byeKey);
  }
  for (size_t x = 0; x < bye->tiebreak_value.size()-1; ++x)
    ASSERT(bye->tiebreak_value[x] == 0);
  //cout << "*bye: " << *bye << BR << endl;
  ASSERT(bye->tiebreak_value.back() == -1);  // coin flip

  //for (PRMiter i = prm.begin(); i != prm.end(); ++i) {
    //const PlayerResult &p = i->second;
    //cout << p << BR << endl;
  //}
  //cout << "end TiebreakCalculation()"BR << endl;
}
